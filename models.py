"""
models.py — All Pydantic data models for SpecGuardian++.
Single source of truth for types shared across env, rewards, grader, and agents.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def numeric(self) -> float:
        return {"low": 0.0, "medium": 0.33, "high": 0.66, "critical": 1.0}[self.value]


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def numeric(self) -> float:
        return {"low": 0.0, "medium": 0.33, "high": 0.66, "critical": 1.0}[self.value]


class Verdict(str, Enum):
    HEALTHY = "system_healthy"
    DEGRADED = "system_degraded"
    CRITICAL = "system_critical"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class File(BaseModel, frozen=True):
    id: str
    name: str
    summary: str
    risk_level: RiskLevel

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("File id must not be empty")
        return v


class Issue(BaseModel, frozen=True):
    id: str
    description: str
    severity: Severity
    fixable: bool
    related_files: List[str]

    @field_validator("related_files")
    @classmethod
    def at_least_one_file(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Issue must reference at least one file")
        return v


class Patch(BaseModel, frozen=True):
    id: str
    description: str
    fixes_issue_id: str
    risk: RiskLevel
    irreversible: bool
    forbidden: bool = False

    @model_validator(mode="after")
    def forbidden_implies_high_risk(self) -> "Patch":
        if self.forbidden and self.risk not in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            raise ValueError("Forbidden patches must carry HIGH or CRITICAL risk")
        return self


# ---------------------------------------------------------------------------
# Episode tracking
# ---------------------------------------------------------------------------

class ActionRecord(BaseModel, frozen=True):
    step: int
    action: str
    args: Dict[str, Any]
    reward: float
    info: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    """Mutable episode state. Only SpecGuardianEnv should mutate this."""

    # Scenario definition (immutable during episode)
    files: List[File]
    issues: List[Issue]
    patches: List[Patch]

    # Episode config
    max_steps: int = Field(default=10, ge=1)

    # Agent progress (mutated by env)
    inspected_files: List[str] = Field(default_factory=list)
    detected_issues: List[str] = Field(default_factory=list)
    applied_patches: List[str] = Field(default_factory=list)
    escalated_issues: List[str] = Field(default_factory=list)
    action_history: List[ActionRecord] = Field(default_factory=list)

    # Episode metrics
    step_count: int = 0
    total_reward: float = 0.0
    system_integrity: float = Field(default=1.0, ge=0.0, le=1.0)

    # Terminal state
    done: bool = False
    verdict: Optional[Verdict] = None
    verdict_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.step_count)

    @property
    def is_terminal(self) -> bool:
        return self.done or self.step_count >= self.max_steps


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

class Observation(BaseModel, frozen=True):
    """What an agent sees at each step — no hidden task info."""

    step: int
    steps_remaining: int
    system_integrity: float
    files: List[Dict[str, Any]]
    issues: List[Dict[str, Any]]
    patches: List[Dict[str, Any]]
    inspected_files: List[str]
    detected_issues: List[str]
    applied_patches: List[str]
    escalated_issues: List[str]
    last_action: Optional[ActionRecord]
    done: bool


class RewardEvent(BaseModel, frozen=True):
    """Single reward signal with full attribution."""

    action: str
    args: Dict[str, Any]
    reward: float
    tags: List[str] = Field(default_factory=list)
    detail: str = ""


class DimensionScore(BaseModel, frozen=True):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    weighted: float

    @classmethod
    def make(cls, name: str, score: float, weight: float) -> "DimensionScore":
        score = max(0.0, min(1.0, score))
        return cls(name=name, score=round(score, 4), weight=weight, weighted=round(score * weight, 4))


class GradeResult(BaseModel, frozen=True):
    """Full grading output for a completed episode."""

    task_id: str
    difficulty: str
    score: float = Field(ge=0.0, le=1.0)
    success: bool
    letter_grade: str
    dimensions: List[DimensionScore]
    total_reward: float
    steps_used: int
    max_steps: int
    system_integrity: float
    verdict_given: Optional[str]
    verdict_correct: bool
    forbidden_patches_applied: List[str]
    summary: str

    @classmethod
    def letter(cls, score: float) -> str:
        if score >= 0.95:
            return "S"
        if score >= 0.85:
            return "A"
        if score >= 0.75:
            return "B"
        if score >= 0.65:
            return "C"
        if score >= 0.50:
            return "D"
        return "F"
