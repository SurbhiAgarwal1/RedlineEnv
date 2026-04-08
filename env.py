"""
env.py — SpecGuardianEnv: the core episode loop.

Design principles:
  - State mutation is centralised here. No other module writes to State.
  - Observations are typed (Observation model) — agents get a clean interface.
  - Rewards are computed BEFORE side effects so reward reflects the pre-action state.
  - Terminal conditions: submit_verdict OR step budget exhausted.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple, TYPE_CHECKING

from models import ActionRecord, Observation, RewardEvent, State
from rewards import compute_reward

if TYPE_CHECKING:
    from tasks import Task


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SpecGuardianEnv:
    """
    Gym-style environment for SpecGuardian++ evaluation episodes.

    Usage:
        env = SpecGuardianEnv(task)
        obs = env.reset()
        while not obs.done:
            action, args = agent.act(obs)
            obs, reward, done, event = env.step(action, args)
        result = grade_episode(env.state, env.task)
    """

    def __init__(self, task: "Task") -> None:
        self.task = task
        self.state: State = task.build_state()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset episode and return initial observation."""
        self.state = self.task.build_state()
        return self._observe()

    def step(
        self,
        action: str,
        args: Dict[str, Any],
    ) -> Tuple[Observation, float, bool, RewardEvent]:
        """
        Execute one action.

        Returns:
            obs     — typed Observation of the new state
            reward  — scalar reward for this step
            done    — True if episode has ended
            event   — RewardEvent with full attribution
        """
        if self.state.done:
            noop = RewardEvent(action=action, args=args, reward=0.0, tags=["noop"], detail="Episode already done")
            return self._observe(), 0.0, True, noop

        # 1. Compute reward (pre-mutation)
        reward, event = compute_reward(self.state, action, args, self.task)

        # 2. Apply side effects
        _apply_action(self.state, action, args)

        # 3. Advance step counter
        self.state.step_count += 1
        self.state.total_reward = round(self.state.total_reward + reward, 4)

        # 4. Record history
        record = ActionRecord(
            step=self.state.step_count,
            action=action,
            args=args,
            reward=reward,
            info={"tags": event.tags, "detail": event.detail},
        )
        self.state.action_history.append(record)

        # 5. Check terminal conditions
        if action == "submit_verdict" or self.state.step_count >= self.state.max_steps:
            self.state.done = True

        return self._observe(), reward, self.state.done, event

    def observation(self) -> Observation:
        """Return current observation without advancing state."""
        return self._observe()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self) -> Observation:
        s = self.state
        return Observation(
            step=s.step_count,
            steps_remaining=s.steps_remaining,
            system_integrity=s.system_integrity,
            files=[f.model_dump() for f in s.files],
            issues=[i.model_dump() for i in s.issues],
            patches=[p.model_dump() for p in s.patches],
            inspected_files=list(s.inspected_files),
            detected_issues=list(s.detected_issues),
            applied_patches=list(s.applied_patches),
            escalated_issues=list(s.escalated_issues),
            last_action=s.action_history[-1] if s.action_history else None,
            done=s.done,
        )


# ---------------------------------------------------------------------------
# Action side-effects (state mutation — ONLY called from env.step)
# ---------------------------------------------------------------------------

def _apply_action(s: State, action: str, args: Dict[str, Any]) -> None:
    """
    Apply the deterministic side-effects of an action to the episode state.
    Must remain in sync with rewards.py logic.
    """
    if action == "inspect_file":
        fid = args.get("file_id")
        if fid and fid not in s.inspected_files:
            s.inspected_files.append(fid)

    elif action == "mark_issue":
        iid = args.get("issue_id")
        issue = next((i for i in s.issues if i.id == iid), None)
        if issue and iid not in s.detected_issues:
            s.detected_issues.append(iid)

    elif action == "escalate_issue":
        iid = args.get("issue_id")
        issue = next((i for i in s.issues if i.id == iid), None)
        if issue and iid not in s.escalated_issues:
            s.escalated_issues.append(iid)

    elif action == "apply_patch":
        pid = args.get("patch_id")
        patch = next((p for p in s.patches if p.id == pid), None)
        if patch and pid not in s.applied_patches:
            s.applied_patches.append(pid)
            # Integrity damage
            if patch.forbidden:
                s.system_integrity = round(max(0.0, s.system_integrity - 0.5), 4)
            elif patch.irreversible:
                s.system_integrity = round(max(0.0, s.system_integrity - 0.3), 4)
            elif patch.risk.value == "high":
                s.system_integrity = round(max(0.0, s.system_integrity - 0.1), 4)

    elif action == "submit_verdict":
        from models import Verdict
        verdict_str = args.get("verdict")
        try:
            s.verdict = Verdict(verdict_str)
        except ValueError:
            s.verdict = None  # Invalid verdict string — penalised by reward
        confidence = args.get("confidence", 0.5)
        s.verdict_confidence = float(max(0.0, min(1.0, confidence)))

    # preview_patch and unknown actions: no state mutation
