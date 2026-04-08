"""
agent.py — Agent interface and built-in agent implementations.

Implement BaseAgent to plug in any policy (LLM, RL, rule-based, random).
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from models import Observation, Verdict

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """
    Interface contract for all SpecGuardian++ agents.

    An agent receives a typed Observation and returns (action, args).
    It must be stateless between calls — all relevant context is in the obs.
    """

    @abc.abstractmethod
    def act(self, obs: Observation) -> Tuple[str, Dict[str, Any]]:
        """
        Decide the next action given the current observation.

        Args:
            obs: Current environment observation (immutable).

        Returns:
            (action_name, args_dict)
        """

    def reset(self) -> None:
        """Called at the start of each episode. Override if agent is stateful."""


# ---------------------------------------------------------------------------
# Rule-based agent (correct, deterministic)
# ---------------------------------------------------------------------------

class RuleBasedAgent(BaseAgent):
    """
    Deterministic expert agent following a principled pipeline:

      1. Inspect all files (critical/high risk first).
      2. Preview risky/unfamiliar patches exactly once before deciding.
      3. Mark all issues.
      4. Apply safe patches (non-forbidden, non-irreversible, low/medium risk).
      5. Escalate non-fixable issues.
      6. Submit a verdict derived from detected issue severities.
    """

    def reset(self) -> None:
        self._previewed: set = set()

    def __init__(self) -> None:
        self._previewed: set = set()

    def act(self, obs: Observation) -> Tuple[str, Dict[str, Any]]:
        inspected = set(obs.inspected_files)
        detected = set(obs.detected_issues)
        escalated = set(obs.escalated_issues)
        applied = set(obs.applied_patches)

        files = obs.files
        issues = obs.issues
        patches = obs.patches

        # --- 1. Inspect files (critical/high risk first) ---
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        uninspected = [f for f in files if f["id"] not in inspected]
        uninspected.sort(key=lambda f: risk_order.get(f["risk_level"], 9))
        if uninspected:
            return "inspect_file", {"file_id": uninspected[0]["id"]}

        # --- 2. Preview dangerous patches exactly once ---
        for patch in patches:
            pid = patch["id"]
            if pid in applied or pid in self._previewed:
                continue
            if patch["forbidden"] or patch["irreversible"] or patch["risk"] in ("high", "critical"):
                self._previewed.add(pid)
                return "preview_patch", {"patch_id": pid}

        # --- 3. Mark all issues ---
        for issue in issues:
            if issue["id"] not in detected:
                return "mark_issue", {"issue_id": issue["id"]}

        # --- 4. Apply safe patches only ---
        for patch in patches:
            pid = patch["id"]
            if pid in applied:
                continue
            if patch["forbidden"] or patch["irreversible"]:
                continue
            if patch["risk"] in ("high", "critical"):
                continue
            issue = next((i for i in issues if i["id"] == patch["fixes_issue_id"]), None)
            if issue and issue["fixable"] and issue["id"] in detected:
                return "apply_patch", {"patch_id": pid}

        # --- 5. Escalate non-fixable issues ---
        for issue in issues:
            iid = issue["id"]
            if not issue["fixable"] and iid in detected and iid not in escalated:
                return "escalate_issue", {"issue_id": iid}

        # --- 6. Submit verdict ---
        return "submit_verdict", {
            "verdict": _derive_verdict(issues, detected, escalated),
            "confidence": 0.90,
        }


def _derive_verdict(issues: List[dict], detected: set, escalated: set) -> str:
    """
    Determine the correct verdict from detected issue severities.

    Logic:
      - Any CRITICAL or HIGH issue that is non-fixable (escalated) → system_critical
      - Any HIGH issue that could not be fixed                      → system_critical
      - Any detected issue remains (fixable but not critical)       → system_degraded
      - No issues detected                                          → system_healthy

    Key nuance: a CRITICAL issue that IS fixable and HAS been escalated
    means it cannot be resolved → system_critical.
    A CRITICAL issue that is fixable (patchable) → system_degraded after patching.
    """
    detected_issues = [i for i in issues if i["id"] in detected]

    # Non-fixable issues that were escalated = unresolvable
    has_unresolvable = any(
        not i["fixable"] and (i["severity"] in ("critical", "high"))
        for i in detected_issues
    )

    # Non-fixable issues of any severity that were escalated
    has_any_escalated = any(
        not i["fixable"] and i["id"] in escalated
        for i in detected_issues
    )

    if has_unresolvable or has_any_escalated:
        # Only critical if severity warrants it
        critical_unresolved = any(
            not i["fixable"] and i["severity"] in ("critical", "high")
            for i in detected_issues
        )
        if critical_unresolved:
            return Verdict.CRITICAL.value

    if detected_issues:
        return Verdict.DEGRADED.value
    return Verdict.HEALTHY.value


# ---------------------------------------------------------------------------
# Deterministic "bad" agent (for testing traps)
# ---------------------------------------------------------------------------

class TrapAgent(BaseAgent):
    """
    An agent that deliberately applies forbidden/irreversible patches.
    Used to verify that the environment penalises dangerous behaviour correctly.
    """

    def act(self, obs: Observation) -> Tuple[str, Dict[str, Any]]:
        applied = set(obs.applied_patches)

        # Apply all patches including forbidden/irreversible ones
        for patch in obs.patches:
            if patch["id"] not in applied:
                return "apply_patch", {"patch_id": patch["id"]}

        return "submit_verdict", {"verdict": Verdict.HEALTHY.value, "confidence": 0.99}


# ---------------------------------------------------------------------------
# Minimal agent (for step-budget and efficiency testing)
# ---------------------------------------------------------------------------

class MinimalAgent(BaseAgent):
    """
    Agent that skips straight to submitting a verdict.
    Tests what happens when an agent acts without understanding the system.
    """

    def act(self, obs: Observation) -> Tuple[str, Dict[str, Any]]:
        return "submit_verdict", {"verdict": Verdict.HEALTHY.value, "confidence": 0.5}
