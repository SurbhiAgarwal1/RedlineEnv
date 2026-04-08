"""
inference.py — Episode runner and evaluation suite for SpecGuardian++.

Entrypoints:
  run_episode(task, agent)   → GradeResult
  run_suite(tasks, agent)    → List[GradeResult]
  __main__                   → runs all tasks × all built-in agents, prints report
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Type, TYPE_CHECKING

from agent import BaseAgent, MinimalAgent, RuleBasedAgent, TrapAgent
from env import SpecGuardianEnv
from graders import grade_episode
from models import GradeResult, Observation
from tasks import TASKS, Task, get_task

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_episode(
    task: Task,
    agent: BaseAgent,
    verbose: bool = True,
) -> GradeResult:
    """
    Run a single episode to completion and return a GradeResult.

    Args:
        task:    Task instance to evaluate on.
        agent:   Agent to run. Must implement BaseAgent.act().
        verbose: If True, print step-by-step trace to stdout.

    Returns:
        GradeResult with full scoring breakdown.
    """
    env = SpecGuardianEnv(task)
    agent.reset()
    obs: Observation = env.reset()

    if verbose:
        _print_header(task)

    while not obs.done:
        action, args = agent.act(obs)
        obs, reward, done, event = env.step(action, args)

        if verbose:
            _print_step(env.state.step_count, action, args, reward, event)

        if done:
            break

    result = grade_episode(env.state, task)

    if verbose:
        _print_result(result)

    return result


def run_suite(
    tasks: Optional[Dict[str, Task]] = None,
    agent: Optional[BaseAgent] = None,
    verbose: bool = True,
) -> List[GradeResult]:
    """
    Run all tasks with the given agent and return all GradeResults.

    Args:
        tasks:   Dict of task_id → Task. Defaults to all built-in tasks.
        agent:   Agent to evaluate. Defaults to RuleBasedAgent.
        verbose: Print per-episode traces.

    Returns:
        List of GradeResults in task order.
    """
    tasks = tasks or TASKS
    agent = agent or RuleBasedAgent()
    return [run_episode(t, agent, verbose=verbose) for t in tasks.values()]


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def _print_header(task: Task) -> None:
    w = 64
    print(f"\n{'═' * w}")
    print(f"  Task : {task.id}  ({task.difficulty.upper()})")
    print(f"  Steps: max {task.build_state().max_steps}")
    print(f"{'═' * w}")


def _print_step(step: int, action: str, args: dict, reward: float, event) -> None:
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    tags_str = " ".join(f"[{t}]" for t in event.tags)
    print(
        f"  [{step:02d}] {action}({args_str})"
        f"\n       → reward={reward:+.3f}  {tags_str}"
        f"\n         {event.detail}"
    )


def _print_result(result: GradeResult) -> None:
    status = "✓ PASS" if result.success else "✗ FAIL"
    grade_bar = _score_bar(result.score)
    print(f"\n  {'─' * 60}")
    print(f"  {status}  Grade: {result.letter_grade}  Score: {result.score:.2%}  {grade_bar}")
    print()
    for dim in result.dimensions:
        bar = _score_bar(dim.score, width=20)
        print(f"    {dim.name:<12} {dim.score:.2%}  {bar}  (weight {dim.weight:.0%})")
    print()
    print(f"  Reward total : {result.total_reward:+.3f}")
    print(f"  Steps        : {result.steps_used}/{result.max_steps}")
    print(f"  Integrity    : {result.system_integrity:.0%}")
    print(f"  Verdict      : {result.verdict_given or 'none'} {'✓' if result.verdict_correct else '✗'}")
    if result.forbidden_patches_applied:
        print(f"  ⚠  Forbidden : {result.forbidden_patches_applied}")


def _score_bar(score: float, width: int = 24) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _print_suite_summary(results: List[GradeResult], agent_name: str) -> None:
    w = 64
    print(f"\n{'═' * w}")
    print(f"  SUITE SUMMARY — {agent_name}")
    print(f"{'─' * w}")
    total_pass = sum(1 for r in results if r.success)
    for r in results:
        status = "✓" if r.success else "✗"
        print(
            f"  {status} {r.task_id:<12} {r.difficulty:<8}"
            f"  {r.letter_grade}  {r.score:.2%}  reward={r.total_reward:+.4f}"
        )
    print(f"{'─' * w}")
    print(f"  Pass rate: {total_pass}/{len(results)}")
    avg = sum(r.score for r in results) / len(results)
    print(f"  Avg score: {avg:.2%}")
    print(f"{'═' * w}\n")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

AGENTS: Dict[str, BaseAgent] = {
    "RuleBased": RuleBasedAgent(),
    "Trap":      TrapAgent(),
    "Minimal":   MinimalAgent(),
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SpecGuardian++ Evaluation Runner")
    parser.add_argument("--task", default=None, help="Run a single task by ID (e.g. easy_01)")
    parser.add_argument("--agent", default="RuleBased", choices=list(AGENTS), help="Agent to use")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step trace")
    args = parser.parse_args()

    agent = AGENTS[args.agent]
    verbose = not args.quiet

    if args.task:
        task = get_task(args.task)
        run_episode(task, agent, verbose=verbose)
    else:
        print(f"\nRunning full suite with agent: {args.agent}")
        results = run_suite(agent=agent, verbose=verbose)
        _print_suite_summary(results, args.agent)
