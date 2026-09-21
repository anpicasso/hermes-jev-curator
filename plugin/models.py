"""Small, serializable contracts shared by the Jev curator modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


MODES = frozenset({"off", "observe", "guard", "apply"})
RELATIONS = (
    "duplicate",
    "a_subset_of_b",
    "b_subset_of_a",
    "same_class",
    "complementary",
    "conflict",
    "unrelated",
    "insufficient_evidence",
)


@dataclass(frozen=True)
class Settings:
    """Operator settings; semantic thresholds stay versioned in code until measured."""

    mode: str = "observe"
    provider: str = "typesafe"
    base_url: str = ""
    model: str = ""
    key_env: str = ""
    allow_content_egress: bool = False
    timeout_seconds: float = 25.0
    max_requests: int = 50
    max_pairs: int = 100
    top_k: int = 5

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "Settings":
        raw = dict(raw or {})
        raw_mode = raw.get("mode", "observe")
        # YAML 1.1 parses an unquoted `off` as False; preserve the operator's intent.
        mode = "off" if raw_mode is False else str(raw_mode or "observe").strip().lower()
        if mode not in MODES:
            mode = "observe"
        provider = str(raw.get("provider") or "typesafe").strip().lower()
        timeout = _number(raw.get("timeout_seconds"), 25.0, 1.0, 120.0)
        return cls(
            mode=mode,
            provider=provider,
            base_url=str(raw.get("base_url", "") or "").strip(),
            # Hermes reserves the plugin setting root ``model``. Keep the
            # dataclass field conventional, but expose it as ``jev_model``.
            model=str(raw.get("jev_model", raw.get("model", "")) or "").strip(),
            key_env=str(raw.get("key_env", "") or "").strip(),
            allow_content_egress=_boolean(raw.get("allow_content_egress"), False),
            timeout_seconds=timeout,
            max_requests=_integer(raw.get("max_requests"), 50, 1, 500),
            max_pairs=_integer(raw.get("max_pairs"), 100, 1, 2_000),
            top_k=_integer(raw.get("top_k"), 5, 1, 20),
        )


@dataclass(frozen=True)
class SkillArtifact:
    name: str
    path: Path
    description: str
    text: str
    digest: str
    provenance: str = "agent"
    state: str = "active"
    pinned: bool = False
    use_count: int = 0
    last_activity_at: str = ""
    support_files: tuple[str, ...] = ()
    protected_reasons: tuple[str, ...] = ()

    @property
    def protected(self) -> bool:
        return bool(self.protected_reasons)

    def public_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        if not include_text:
            data.pop("text", None)
        return data


@dataclass(frozen=True)
class CandidatePair:
    a: str
    b: str
    a_digest: str
    b_digest: str
    similarity: float
    signals: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return "::".join(sorted((self.a, self.b)))


@dataclass(frozen=True)
class RelationJudgment:
    a: str
    b: str
    a_digest: str
    b_digest: str
    relation: str
    confidence: float
    probabilities: Mapping[str, float]
    coverage: float
    preservation_a_in_b: float
    preservation_b_in_a: float
    conflict: float
    contract_version: str
    raw_model: str = ""
    evidence: str = "whole"

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"unknown relation: {self.relation}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.evidence not in {"whole", "chunked", "unavailable"}:
            raise ValueError(f"unknown evidence: {self.evidence}")

    @property
    def key(self) -> str:
        return "::".join(sorted((self.a, self.b)))


@dataclass(frozen=True)
class MergePlan:
    plan_id: str
    canonical: str
    canonical_digest: str
    absorbed: tuple[str, ...]
    absorbed_digests: Mapping[str, str]
    relation_keys: tuple[str, ...]
    status: str = "proposed"
    blockers: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def applicable(self) -> bool:
        return self.status == "validated" and not self.blockers


def _integer(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _number(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _boolean(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0", ""}:
            return False
    return default
