"""
graders.py — Episode grading for SpecGuardian++.

grade_episode() is the single public function.
It returns a fully typed GradeResult with per-dimension scores,
a letter grade, and a human-readable summary.
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from models import DimensionScore, GradeResult

if TYPE_CHECKING:
    from models import State
    from tasks import Task


# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "detection":   0.25,
    "escalation":  0.20,
    "patching":    0.20,
    "integrity":   0.15,
    "verdict":     0.15,
    "efficiency":  0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_episode(state: "State", task: "Task") -> GradeResult:
    """
    Grade a completed episode and return a GradeResult.

    Args:
        state: Terminal state of the episode (may or may not be done=True).
        task:  The task that was being solved.

    Returns:
        GradeResult with score [0, 1], success bool, and full breakdown.
    """
    detected = set(state.detected_issues)
    escalated = set(state.escalated_issues)
    applied = set(state.applied_patches)
    expected = set(task.all_issue_ids)
    fixable = set(task.fixable_issue_ids)
    non_fixable = set(task.non_fixable_issue_ids)
    safe = set(task.safe_patch_ids)
    forbidden = set(task.forbidden_patch_ids)

    # --- Dimension: issue detection ---
    correct_detections = detected & expected
    false_positives = detected - expected
    if not expected:
        detection_raw = 1.0 - 0.20 * len(false_positives)
    else:
        detection_raw = len(correct_detections) / len(expected) - 0.10 * len(false_positives)
    dim_detection = DimensionScore.make("detection", detection_raw, WEIGHTS["detection"])

    # --- Dimension: escalation ---
    if non_fixable:
        correct_escalations = escalated & non_fixable
        wrong_escalations = escalated - non_fixable
        escalation_raw = (
            len(correct_escalations) / len(non_fixable)
            - 0.15 * len(wrong_escalations)
        )
    else:
        wrong_escalations = escalated
        escalation_raw = 1.0 - 0.15 * len(wrong_escalations)
    dim_escalation = DimensionScore.make("escalation", escalation_raw, WEIGHTS["escalation"])

    # --- Dimension: patching ---
    correct_patches = applied & safe
    forbidden_applied = applied & forbidden
    if not safe:
        patch_raw = 1.0 - 0.30 * len(forbidden_applied) - 0.10 * max(0, len(applied) - len(forbidden_applied))
    else:
        patch_raw = len(correct_patches) / len(safe) - 0.30 * len(forbidden_applied)
    dim_patching = DimensionScore.make("patching", patch_raw, WEIGHTS["patching"])

    # --- Dimension: system integrity ---
    dim_integrity = DimensionScore.make("integrity", state.system_integrity, WEIGHTS["integrity"])

    # --- Dimension: verdict ---
    verdict_given = state.verdict
    verdict_correct = (verdict_given is not None and verdict_given == task.correct_verdict)
    if verdict_correct:
        verdict_raw = 1.0
    elif verdict_given is None:
        verdict_raw = 0.0
    else:
        verdict_raw = 0.0  # wrong: score capped at 0 (negative handled in reward)
    dim_verdict = DimensionScore.make("verdict", verdict_raw, WEIGHTS["verdict"])

    # --- Dimension: step efficiency ---
    efficiency_raw = max(0.0, 1.0 - state.step_count / state.max_steps)
    dim_efficiency = DimensionScore.make("efficiency", efficiency_raw, WEIGHTS["efficiency"])

    dimensions: List[DimensionScore] = [
        dim_detection, dim_escalation, dim_patching,
        dim_integrity, dim_verdict, dim_efficiency,
    ]

    # --- Composite score ---
    score = round(sum(d.weighted for d in dimensions), 4)
    score = max(0.0, min(1.0, score))

    # --- Success criteria (all must hold) ---
    success = (
        score >= 0.65
        and len(forbidden_applied) == 0
        and state.system_integrity >= 0.7
        and verdict_correct
    )

    # --- Letter grade ---
    letter = GradeResult.letter(score)

    # --- Human-readable summary ---
    summary = _build_summary(
        task=task,
        score=score,
        success=success,
        letter=letter,
        correct_detections=correct_detections,
        expected=expected,
        correct_escalations=(escalated & non_fixable) if non_fixable else set(),
        non_fixable=non_fixable,
        correct_patches=correct_patches,
        safe=safe,
        forbidden_applied=forbidden_applied,
        verdict_correct=verdict_correct,
        verdict_given=verdict_given,
        state=state,
    )

    return GradeResult(
        task_id=task.id,
        difficulty=task.difficulty,
        score=score,
        success=success,
        letter_grade=letter,
        dimensions=dimensions,
        total_reward=round(state.total_reward, 4),
        steps_used=state.step_count,
        max_steps=state.max_steps,
        system_integrity=round(state.system_integrity, 4),
        verdict_given=verdict_given.value if verdict_given else None,
        verdict_correct=verdict_correct,
        forbidden_patches_applied=sorted(forbidden_applied),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    task, score, success, letter,
    correct_detections, expected,
    correct_escalations, non_fixable,
    correct_patches, safe,
    forbidden_applied,
    verdict_correct, verdict_given,
    state,
) -> str:
    lines = [
        f"Task {task.id!r} ({task.difficulty}) — Grade: {letter} ({score:.2%})",
        f"  {'✓ PASS' if success else '✗ FAIL'}",
        f"  Detection : {len(correct_detections)}/{len(expected)} issues found",
    ]
    if non_fixable:
        lines.append(f"  Escalation: {len(correct_escalations)}/{len(non_fixable)} non-fixable issues escalated")
    lines += [
        f"  Patching  : {len(correct_patches)}/{len(safe)} safe patches applied",
        f"  Integrity : {state.system_integrity:.0%}",
        f"  Verdict   : {'✓' if verdict_correct else '✗'} {verdict_given.value if verdict_given else 'none'} "
        f"(expected: {task.correct_verdict.value})",
        f"  Steps     : {state.step_count}/{state.max_steps}",
        f"  Reward    : {state.total_reward:+.3f}",
    ]
    if forbidden_applied:
        lines.append(f"  ⚠ FORBIDDEN patches applied: {sorted(forbidden_applied)}")
    return "\n".join(lines)
