"""Deterministic, read-only reporting for Jev curator runs.

Renders scan results — inventory count, candidate pairs with their
deterministic baseline, Jev judgments, merge plans, blockers, errors, and the
per-call latency/token/cost inputs — as Markdown and JSON. Skill bodies are
never rendered and free text is redacted and bounded first. Both artifacts are
written atomically (temp file + ``os.replace``) inside the supplied run
directory; this module never mutates skills, plans, or anything outside that
directory, regardless of mode.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidates import deterministic_relation
from .models import MODES, CandidatePair, MergePlan, RelationJudgment, Settings, SkillArtifact


SCHEMA = "hermes-jev-curator/run-v1"
REPORT_FILENAME = "report.md"
RUN_JSON_FILENAME = "run.json"
DEFAULT_CELL_CHARS = 160

_SAFE_NAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")

@dataclass(frozen=True)
class ScanError:
    """One failure captured during a scan stage; message is redacted at render time."""

    stage: str
    kind: str
    message: str = ""


@dataclass(frozen=True)
class CallRecord:
    """Inputs to latency/token/cost accounting for one Jev call."""

    stage: str
    model: str = ""
    attempts: int = 0
    http_status: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "latency_ms", _safe_float(self.latency_ms))
        object.__setattr__(self, "cost_usd", _safe_float(self.cost_usd))
        for name in ("attempts", "http_status", "prompt_tokens", "completion_tokens", "total_tokens"):
            object.__setattr__(self, name, _safe_int(getattr(self, name)))

    @classmethod
    def from_response(cls, response: Any, *, stage: str, latency_ms: float = 0.0) -> "CallRecord":
        """Build a record from a transport.JevResponse-like object (duck-typed)."""
        usage = getattr(response, "usage", None)
        usage = usage if isinstance(usage, Mapping) else {}
        # ponytail: flat usage keys; add provider paths only when one actually nests them.
        prompt = _safe_int(usage.get("prompt_tokens", usage.get("input_tokens")))
        completion = _safe_int(usage.get("completion_tokens", usage.get("output_tokens")))
        total = _safe_int(usage.get("total_tokens")) or prompt + completion
        return cls(
            stage=str(stage or ""),
            model=str(getattr(response, "model", "") or ""),
            attempts=_safe_int(getattr(response, "attempts", 0)),
            http_status=_safe_int(getattr(response, "http_status", 0)),
            latency_ms=latency_ms,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=_safe_float(usage.get("cost_usd", usage.get("cost", usage.get("total_cost")))),
        )


@dataclass
class RunReport:
    """Collected scan results; __post_init__ fixes a stable order for every section."""

    mode: str = "observe"
    started_at: str = ""
    finished_at: str = ""
    inventory: Sequence[SkillArtifact] = ()
    candidates: Sequence[CandidatePair] = ()
    judgments: Sequence[RelationJudgment] = ()
    plans: Sequence[MergePlan] = ()
    blockers: Sequence[str] = ()
    errors: Sequence[ScanError] = ()
    calls: Sequence[CallRecord] = ()
    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.mode = self.mode if self.mode in MODES else "observe"
        self.inventory = tuple(sorted(self.inventory, key=lambda item: item.name))
        unique: dict[str, CandidatePair] = {}
        for pair in sorted(self.candidates, key=lambda item: (-item.similarity, item.a, item.b)):
            unique.setdefault(pair.key, pair)
        self.candidates = tuple(unique.values())
        self.judgments = tuple(sorted(self.judgments, key=lambda item: item.key))
        self.plans = tuple(sorted(self.plans, key=lambda item: item.plan_id))
        self.blockers = tuple(sorted(str(item) for item in self.blockers))
        self.errors = tuple(sorted(self.errors, key=lambda item: (item.stage, item.kind, item.message)))
        self.calls = tuple(sorted(
            self.calls, key=lambda item: (item.stage, item.model, item.http_status, item.latency_ms)))
        if isinstance(self.settings, Mapping):
            self.settings = Settings.from_mapping(self.settings)


def to_dict(report: RunReport) -> dict[str, Any]:
    """Canonical machine-readable view; both renderers are built from this."""
    inventory = _inventory_rows(report)
    candidates = _candidate_rows(report)
    judgments = _judgment_rows(report)
    plans = _plan_rows(report)
    errors = _error_rows(report)
    calls = _call_rows(report)
    return {
        "schema": SCHEMA,
        "read_only": True,
        "mode": report.mode,
        "started_at": _cell(report.started_at, 64),
        "finished_at": _cell(report.finished_at, 64),
        "summary": _summary(report),
        "inventory": {"count": len(inventory), "skills": inventory},
        "candidates": {"count": len(candidates), "pairs": candidates},
        "judgments": {"count": len(judgments), "by_relation": _relation_counts(report), "pairs": judgments},
        "plans": {"count": len(plans), "applicable": sum(1 for row in plans if row["applicable"]), "items": plans},
        "blockers": [_text(item) for item in report.blockers],
        "errors": {"count": len(errors), "items": errors},
        "calls": {"count": len(calls), "items": calls},
        "settings": _settings_row(report.settings),
    }


def _inventory_rows(report: RunReport) -> list[dict[str, Any]]:
    return [{
        "name": _cell(item.name),
        "digest": _cell(item.digest),
        "state": _cell(item.state),
        "provenance": _cell(item.provenance),
        "pinned": bool(item.pinned),
        "use_count": int(item.use_count),
        "support_files": len(item.support_files),
        "protected_reasons": sorted(_cell(reason) for reason in item.protected_reasons),
    } for item in report.inventory]


def _candidate_rows(report: RunReport) -> list[dict[str, Any]]:
    return [{
        "key": _cell(pair.key),
        "a": _cell(pair.a),
        "b": _cell(pair.b),
        "a_digest": _cell(pair.a_digest),
        "b_digest": _cell(pair.b_digest),
        "similarity": _finite(pair.similarity),
        "signals": [_cell(signal) for signal in pair.signals],
        "baseline": _cell(deterministic_relation(pair)),
    } for pair in report.candidates]


def _judgment_rows(report: RunReport) -> list[dict[str, Any]]:
    return [{
        "key": _cell(item.key),
        "a": _cell(item.a),
        "b": _cell(item.b),
        "relation": _cell(item.relation),
        "confidence": _finite(item.confidence),
        "coverage": _finite(item.coverage),
        "preservation_a_in_b": _finite(item.preservation_a_in_b),
        "preservation_b_in_a": _finite(item.preservation_b_in_a),
        "conflict": _finite(item.conflict),
        "contract_version": _cell(item.contract_version),
        "raw_model": _text(item.raw_model, 80),
        "evidence": _cell(item.evidence, 20),
    } for item in report.judgments]


def _plan_rows(report: RunReport) -> list[dict[str, Any]]:
    return [{
        "plan_id": _cell(plan.plan_id),
        "canonical": _cell(plan.canonical),
        "canonical_digest": _cell(plan.canonical_digest),
        "absorbed": sorted(_cell(name) for name in plan.absorbed),
        "relation_keys": sorted(_cell(key) for key in plan.relation_keys),
        "status": _cell(plan.status),
        "applicable": bool(plan.applicable),
        "blockers": [_text(item) for item in sorted(plan.blockers)],
    } for plan in report.plans]


def _error_rows(report: RunReport) -> list[dict[str, Any]]:
    return [{
        "stage": _cell(item.stage, 60),
        "kind": _cell(item.kind, 60),
        "message": _text(item.message),
    } for item in report.errors]


def _call_rows(report: RunReport) -> list[dict[str, Any]]:
    return [{
        "stage": _cell(item.stage, 60),
        "model": _text(item.model, 80),
        "attempts": item.attempts,
        "http_status": item.http_status,
        "latency_ms": round(item.latency_ms, 3),
        "prompt_tokens": item.prompt_tokens,
        "completion_tokens": item.completion_tokens,
        "total_tokens": item.total_tokens,
        "cost_usd": round(item.cost_usd, 6),
    } for item in report.calls]


def _relation_counts(report: RunReport) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in report.judgments:
        counts[item.relation] = counts.get(item.relation, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _summary(report: RunReport) -> dict[str, Any]:
    calls = report.calls
    return {
        "inventory": len(report.inventory),
        "candidates": len(report.candidates),
        "judgments": len(report.judgments),
        "plans": len(report.plans),
        "plans_applicable": sum(1 for plan in report.plans if plan.applicable),
        "blockers": len(report.blockers),
        "errors": len(report.errors),
        "calls": len(calls),
        "latency_ms": round(sum(call.latency_ms for call in calls), 3),
        "prompt_tokens": sum(call.prompt_tokens for call in calls),
        "completion_tokens": sum(call.completion_tokens for call in calls),
        "total_tokens": sum(call.total_tokens for call in calls),
        "cost_usd": round(sum(call.cost_usd for call in calls), 6),
    }


def _settings_row(settings: Settings | None) -> dict[str, Any] | None:
    """Non-secret settings only: env var names, never values."""
    if settings is None:
        return None
    return {
        "provider": _cell(settings.provider, 40),
        "model": _cell(settings.model, 80),
        "base_url": _cell(settings.base_url),
        "key_env": _cell(settings.key_env, 80),
        "timeout_seconds": round(_safe_float(settings.timeout_seconds), 3),
        "max_requests": settings.max_requests,
        "max_pairs": settings.max_pairs,
        "top_k": settings.top_k,
    }


def render_json(report: RunReport) -> str:
    """Stable JSON text (sorted keys, finite numbers only)."""
    return json.dumps(to_dict(report), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def render_markdown(report: RunReport) -> str:
    """Concise Markdown: summary always, detail sections only when populated."""
    data = to_dict(report)
    summary = data["summary"]
    lines = [f"# Jev curator run — {report.mode}", "", f"- schema: `{SCHEMA}`"]
    if data["started_at"]:
        lines.append(f"- started: {data['started_at']}")
    if data["finished_at"]:
        lines.append(f"- finished: {data['finished_at']}")
    settings = data["settings"]
    if settings:
        lines.append(f"- provider: {_cell(settings['provider'], 40)}")
        lines.append(f"- model: {_cell(settings['model'], 80) or '(route default)'}")
    lines.append("- mutations: none (report only)")

    lines += ["", "## Summary", ""]
    lines += _table(("metric", "value"), (
        ("skills in inventory", str(summary["inventory"])),
        ("candidate pairs", str(summary["candidates"])),
        ("Jev judgments", str(summary["judgments"])),
        ("plans", str(summary["plans"])),
        ("applicable plans", str(summary["plans_applicable"])),
        ("blockers", str(summary["blockers"])),
        ("errors", str(summary["errors"])),
        ("Jev calls", str(summary["calls"])),
        ("latency total (ms)", _num(summary["latency_ms"])),
        ("tokens (prompt / completion / total)",
         f"{summary['prompt_tokens']} / {summary['completion_tokens']} / {summary['total_tokens']}"),
        ("cost (USD)", _num(summary["cost_usd"])),
    ))

    if data["inventory"]["skills"]:
        lines += ["", "## Inventory", ""]
        lines += _table(("skill", "state", "provenance", "digest", "protected"), [
            (
                _cell(row["name"]),
                _cell(row["state"], 40),
                _cell(row["provenance"], 40),
                _cell(row["digest"], 12),
                _cell(", ".join(row["protected_reasons"])) or "—",
            )
            for row in data["inventory"]["skills"]
        ])

    if data["candidates"]["pairs"]:
        lines += ["", "## Candidates", ""]
        lines += _table(("pair", "similarity", "signals", "baseline"), [
            (
                _cell(row["key"]),
                _num(row["similarity"]),
                _cell(", ".join(row["signals"])) or "—",
                _cell(row["baseline"], 40),
            )
            for row in data["candidates"]["pairs"]
        ])

    if data["judgments"]["pairs"]:
        lines += ["", "## Jev judgments", ""]
        lines += _table(("pair", "relation", "confidence", "coverage", "conflict"), [
            (
                _cell(row["key"]),
                _cell(row["relation"], 40),
                _num(row["confidence"]),
                _num(row["coverage"]),
                _num(row["conflict"]),
            )
            for row in data["judgments"]["pairs"]
        ])

    if data["plans"]["items"]:
        lines += ["", "## Plans", ""]
        lines += _table(("plan", "canonical", "absorbed", "status", "applicable", "blockers"), [
            (
                _cell(row["plan_id"], 60),
                _cell(row["canonical"]),
                _cell(", ".join(row["absorbed"])) or "—",
                _cell(row["status"], 40),
                "yes" if row["applicable"] else "no",
                _cell("; ".join(row["blockers"])) or "—",
            )
            for row in data["plans"]["items"]
        ])

    if data["blockers"]:
        lines += ["", "## Blockers", ""]
        lines += [f"- {_cell(item)}" for item in data["blockers"]]

    if data["errors"]["items"]:
        lines += ["", "## Errors", ""]
        lines += _table(("stage", "kind", "message"), [
            (_cell(row["stage"], 60), _cell(row["kind"], 60), _cell(row["message"]))
            for row in data["errors"]["items"]
        ])

    if data["calls"]["items"]:
        lines += ["", "## Jev calls", ""]
        lines += _table(("stage", "model", "attempts", "status", "latency ms", "tokens p/c/t", "cost usd"), [
            (
                _cell(row["stage"], 60),
                _cell(row["model"], 80) or "—",
                str(row["attempts"]),
                str(row["http_status"]),
                _num(row["latency_ms"]),
                f"{row['prompt_tokens']}/{row['completion_tokens']}/{row['total_tokens']}",
                _num(row["cost_usd"]),
            )
            for row in data["calls"]["items"]
        ])

    return "\n".join(lines) + "\n"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def safe_name(name: str) -> str:
    """Return ``name`` when it is a safe single-component filename, else raise."""
    candidate = str(name or "").strip()
    if not _SAFE_NAME.match(candidate):
        raise ValueError(f"unsafe report filename: {name!r}")
    return candidate


def run_paths(run_dir: Path) -> tuple[Path, Path]:
    """Return the (report.md, run.json) paths inside ``run_dir``; names are validated."""
    directory = Path(run_dir)
    return directory / safe_name(REPORT_FILENAME), directory / safe_name(RUN_JSON_FILENAME)


def write_report(report: RunReport, run_dir: Path) -> tuple[Path, Path]:
    """Atomically write report.md and run.json under ``run_dir``; returns their paths."""
    directory = Path(run_dir)
    if directory.is_symlink():
        raise ValueError("refusing a symlinked report directory")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not directory.is_dir():
        raise ValueError("report path is not a directory")
    report_path, json_path = run_paths(directory)
    _write_atomic(json_path, render_json(report))
    _write_atomic(report_path, render_markdown(report))
    return report_path, json_path


def _write_atomic(path: Path, text: str) -> None:
    if path.is_symlink():
        raise ValueError("refusing to replace a symlinked report file")
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _cell(value: Any, limit: int = DEFAULT_CELL_CHARS) -> str:
    """Redact, collapse whitespace, escape Markdown separators, and bound length."""
    from .state import redact_text
    text = " ".join(redact_text("" if value is None else str(value)).split())
    if len(text) > limit:
        text = text[: max(1, limit - 1)] + "…"
    return text.replace("|", "\\|")


def _text(value: Any, limit: int = DEFAULT_CELL_CHARS) -> str:
    """Compatibility alias: every cell now passes through the same redaction choke point."""
    return _cell(value, limit)


def _num(value: Any) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:.6g}"


def _finite(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default
