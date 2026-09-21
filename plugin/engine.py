"""Orchestrates inventory, deterministic candidates, Jev judgments, and safe execution."""

from __future__ import annotations

import json
import shlex
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Iterable, Mapping

from .candidates import deterministic_relation, generate_candidates
from .inventory import collect_inventory, digests_match
from .models import CandidatePair, MergePlan, RelationJudgment, Settings, SkillArtifact
from .questions import CONTRACT_VERSION, has_truncation, pair_state, relation_questions
from .transport import request


_MAX_WORKERS = 4


class CuratorEngine:
    def __init__(self, settings: Settings, ctx: Any = None):
        self.settings = settings
        self.ctx = ctx

    def status(self) -> dict[str, Any]:
        artifacts = collect_inventory()
        return {
            "ok": True,
            "mode": self.settings.mode,
            "managed_skills": len(artifacts),
            "protected_skills": sum(item.protected for item in artifacts),
            "provider": self.settings.provider,
            "network_enabled": self.settings.mode != "off" and self.settings.allow_content_egress,
        }

    def scan(self, names: Iterable[str] | None = None, *, use_jev: bool = True) -> dict[str, Any]:
        started = time.monotonic()
        artifacts = collect_inventory()
        selected = {str(name) for name in (names or []) if str(name)}
        pairs = generate_candidates(
            artifacts, top_k=self.settings.top_k, max_pairs=self.settings.max_pairs)
        if selected:
            pairs = [pair for pair in pairs if selected.intersection((pair.a, pair.b))]
        baseline = {pair.key: deterministic_relation(pair) for pair in pairs}
        judgments: list[RelationJudgment] = []
        errors: list[dict[str, str]] = []
        cache_hits = 0
        jev_enabled = use_jev and self.settings.mode != "off" and self.settings.allow_content_egress
        if jev_enabled:
            from .state import cached_relation, load_relation_cache, remember_relation, save_relation_cache
            from .transport import resolve_route

            budgeted = pairs[: self.settings.max_requests]
            by_name = {item.name: item for item in artifacts}
            cache = load_relation_cache()
            try:
                model = resolve_route(self.settings).model
            except Exception as exc:
                # A bad operator setting is a failed judgment run, not a crashed curator.
                errors.append({"pair": "", "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
                model = ""
                budgeted = []
            pending: list[CandidatePair] = []
            for pair in budgeted:
                cached = cached_relation(
                    cache, pair.a_digest, pair.b_digest,
                    contract_version=CONTRACT_VERSION, model=model)
                if cached is None:
                    pending.append(pair)
                else:
                    judgments.append(cached)
                    cache_hits += 1
            with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, max(1, len(pending)))) as pool:
                futures = {
                    pool.submit(self._judge_pair, pair, by_name[pair.a], by_name[pair.b]): pair
                    for pair in pending
                }
                for future in as_completed(futures):
                    pair = futures[future]
                    try:
                        judgment = future.result()
                        judgments.append(judgment)
                        remember_relation(cache, judgment, model=model)
                    except Exception as exc:
                        errors.append({"pair": pair.key, "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
            if pending:
                try:
                    save_relation_cache(cache)
                except Exception as exc:
                    errors.append({"pair": "", "error": f"relation cache write failed: {type(exc).__name__}"})
        judgments.sort(key=lambda item: (item.a, item.b))
        return {
            "ok": not errors,
            "mode": self.settings.mode,
            "contract_version": CONTRACT_VERSION,
            "inventory": [item.public_dict() for item in artifacts],
            "candidates": [asdict(pair) for pair in pairs],
            "baseline": baseline,
            "judgments": [asdict(item) for item in judgments],
            "cache_hits": cache_hits,
            "content_egress_enabled": self.settings.allow_content_egress,
            "jev_skipped": ("content egress is disabled" if use_jev and self.settings.mode != "off"
                            and not self.settings.allow_content_egress else ""),
            "errors": errors,
            "latency_ms": int((time.monotonic() - started) * 1000),
        }

    def review(self, first: str, second: str | None = None) -> dict[str, Any]:
        if self.settings.mode == "off":
            return {"ok": False, "error": "Jev requests are disabled while mode is off"}
        if not self.settings.allow_content_egress:
            return {"ok": False, "error": "content egress is disabled; set allow_content_egress: true"}
        artifacts = collect_inventory(include_unmanaged=True)
        by_name = {item.name: item for item in artifacts}
        if first not in by_name:
            return {"ok": False, "error": f"unknown skill: {first}"}
        if second:
            if second not in by_name:
                return {"ok": False, "error": f"unknown skill: {second}"}
            pair = _pair(by_name[first], by_name[second])
            try:
                return {"ok": True, "judgment": asdict(self._judge_pair(pair, by_name[pair.a], by_name[pair.b]))}
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:240]}"}
        return self.scan(names=[first], use_jev=True)

    def build_plans(self, scan: Mapping[str, Any], *, max_plans: int = 5) -> list[MergePlan]:
        from .graph import build_plans
        artifacts = collect_inventory()
        judgments = [_judgment_from_mapping(item) for item in scan.get("judgments", [])]
        return build_plans(artifacts, judgments, max_plans=max_plans)

    def apply(self, plans: Iterable[MergePlan]) -> dict[str, Any]:
        """Archive only validated direct-containment sources into an existing canonical skill."""
        if self.settings.mode != "apply":
            return {"ok": False, "error": "apply requires plugins.entries.jev-curator.settings.mode: apply"}
        if self.ctx is None:
            return {"ok": False, "error": "plugin context is unavailable"}
        selected = [plan for plan in plans if plan.applicable]
        if not selected:
            return {"ok": True, "applied": [], "message": "no validated plans"}
        absorbed = [name for plan in selected for name in plan.absorbed]
        canonicals = {plan.canonical for plan in selected}
        for plan in selected:
            expected_edges = {"::".join(sorted((plan.canonical, name))) for name in plan.absorbed}
            if (not plan.canonical or not plan.canonical_digest or not plan.absorbed
                    or any(not name or not plan.absorbed_digests.get(name) for name in plan.absorbed)
                    or not expected_edges.issubset(set(plan.relation_keys))):
                return {"ok": False, "error": (
                    f"plan {plan.plan_id!r} lacks complete hash-bound direct-edge evidence; rescan required")}
        if len(absorbed) != len(set(absorbed)) or canonicals.intersection(absorbed):
            return {"ok": False, "error": "plans overlap or form a consolidation chain; rescan required"}
        # Hermes currently replays approved skill writes without restoring the
        # original background-review provenance. A staged delete could then be
        # replayed as a foreground hard delete instead of a recoverable archive.
        # Refuse before creating any pending mutation until core preserves it.
        try:
            from tools.write_approval import SKILLS, discard_pending, write_approval_enabled
            if write_approval_enabled(SKILLS):
                return {"ok": False, "error": (
                    "skills.write_approval is enabled; apply is refused because approved "
                    "replay does not preserve curator provenance")}
        except Exception as exc:
            return {"ok": False, "error": f"could not verify skills.write_approval: {type(exc).__name__}"}
        expected = {
            plan.canonical: plan.canonical_digest
            for plan in selected
        }
        for plan in selected:
            expected.update(plan.absorbed_digests)
        current = collect_inventory(include_unmanaged=True, names=set(expected))
        if not digests_match(expected, current):
            return {"ok": False, "error": "skill package changed after judgment; rescan required"}
        protected = sorted(item.name for item in current if item.name in expected and item.protected)
        if protected:
            return {"ok": False, "error": (
                f"protected skill cannot be applied: {', '.join(protected)}; rescan required")}
        try:
            from agent.curator_backup import snapshot_skills
            snapshot = snapshot_skills(reason="pre-jev-curator-apply")
        except Exception as exc:
            return {"ok": False, "error": f"snapshot failed: {type(exc).__name__}"}
        if snapshot is None:
            return {"ok": False, "error": "snapshot unavailable; refusing destructive apply"}

        results: list[dict[str, Any]] = []
        staged: list[dict[str, Any]] = []
        verified_archives: list[str] = []
        snapshot_name = getattr(snapshot, "name", str(snapshot))

        def failed(error: str, **extra: Any) -> dict[str, Any]:
            return {
                "ok": False,
                "snapshot": snapshot_name,
                "applied": results,
                "staged": staged,
                "recovery": {
                    "snapshot": snapshot_name,
                    "restore_commands": [
                        f"hermes curator restore {shlex.quote(name)}"
                        for name in verified_archives
                    ],
                },
                "error": error,
                **extra,
            }

        from tools.skill_provenance import (
            BACKGROUND_REVIEW,
            reset_current_write_origin,
            reset_review_attended,
            set_current_write_origin,
            set_review_attended,
        )
        origin_token = set_current_write_origin(BACKGROUND_REVIEW)
        attended_token = set_review_attended(True)
        try:
            for plan in selected:
                for source in plan.absorbed:
                    pair_expected = {
                        source: plan.absorbed_digests[source],
                        plan.canonical: plan.canonical_digest,
                    }
                    try:
                        latest = collect_inventory(
                            include_unmanaged=True, names=set(pair_expected))
                    except Exception as exc:
                        return failed(
                            f"pre-write inventory verification failed: {type(exc).__name__}")
                    if not digests_match(pair_expected, latest):
                        return failed(
                            f"skill package changed immediately before archiving {source!r}; rescan required")
                    if any(item.protected for item in latest if item.name in pair_expected):
                        return failed(f"protected skill detected before archiving {source!r}; rescan required")
                    try:
                        raw = self.ctx.dispatch_tool(
                            "skill_manage", {"action": "delete", "name": source,
                                             "absorbed_into": plan.canonical})
                    except Exception as exc:
                        return failed(
                            f"skill_manage raised for {source!r}: {type(exc).__name__}; stopping")
                    result = _json_result(raw)
                    row = {"plan_id": plan.plan_id, "skill": source, "result": result}
                    if isinstance(result, dict) and result.get("staged") is True:
                        pending_id = str(result.get("pending_id") or "")
                        try:
                            discarded = bool(pending_id) and discard_pending(SKILLS, pending_id)
                        except Exception:
                            discarded = False
                        row["pending_discarded"] = discarded
                        staged.append(row)
                        detail = ("staged write was discarded"
                                  if discarded else "staged write could not be discarded; discard it manually")
                        return failed(f"skill write was staged rather than applied; {detail}; stopping")
                    results.append(row)
                    if not _result_ok(result):
                        return failed(
                            f"skill_manage failed for {source!r}; remaining plans were not applied")
                    if result.get("_archived") is not True:
                        return failed(
                            f"skill_manage did not confirm a recoverable archive for {source!r}; stopping")
                    try:
                        post = {item.name: item for item in collect_inventory(
                            include_unmanaged=True, names={source, plan.canonical})}
                    except Exception as exc:
                        return failed(
                            f"post-write inventory verification failed: {type(exc).__name__}")
                    if source in post:
                        return failed(f"archived skill {source!r} is still present; stopping")
                    verified_archives.append(source)
                    canonical = post.get(plan.canonical)
                    if canonical is None or canonical.digest != plan.canonical_digest:
                        return failed(
                            f"canonical changed after archiving {source!r}; restore archived skills")
        finally:
            reset_review_attended(attended_token)
            reset_current_write_origin(origin_token)
        return {"ok": all(_result_ok(item["result"]) for item in results),
                "snapshot": snapshot_name, "applied": results,
                "staged": staged}

    def _judge_pair(
        self, pair: CandidatePair, a: SkillArtifact, b: SkillArtifact,
    ) -> RelationJudgment:
        state = pair_state(a, b, self.settings.max_state_chars)
        if has_truncation(state):
            return RelationJudgment(
                a=a.name, b=b.name, a_digest=a.digest, b_digest=b.digest,
                relation="insufficient_evidence", confidence=1.0,
                probabilities={"insufficient_evidence": 1.0}, coverage=0.0,
                preservation_a_in_b=0.0, preservation_b_in_a=0.0, conflict=0.0,
                contract_version=CONTRACT_VERSION,
            )
        response = request(state, relation_questions(), self.settings)
        relation = response.answers["relation"]
        return RelationJudgment(
            a=a.name,
            b=b.name,
            a_digest=a.digest,
            b_digest=b.digest,
            relation=str(relation["choice"]),
            confidence=float(relation["confidence"]),
            probabilities=dict(relation.get("probabilities") or {}),
            coverage=float(response.answers["coverage"]["noul"]),
            preservation_a_in_b=float(response.answers["a_in_b"]["noul"]),
            preservation_b_in_a=float(response.answers["b_in_a"]["noul"]),
            conflict=float(response.answers["conflict"]["noul"]),
            contract_version=CONTRACT_VERSION,
            raw_model=response.model,
        )


def _pair(a: SkillArtifact, b: SkillArtifact) -> CandidatePair:
    first, second = sorted((a, b), key=lambda item: item.name)
    return CandidatePair(first.name, second.name, first.digest, second.digest, 1.0, ("explicit",))


def _judgment_from_mapping(raw: Mapping[str, Any]) -> RelationJudgment:
    return RelationJudgment(**{
        "a": str(raw["a"]), "b": str(raw["b"]),
        "a_digest": str(raw["a_digest"]), "b_digest": str(raw["b_digest"]),
        "relation": str(raw["relation"]), "confidence": float(raw["confidence"]),
        "probabilities": dict(raw.get("probabilities") or {}),
        "coverage": float(raw.get("coverage") or 0.0),
        "preservation_a_in_b": float(raw.get("preservation_a_in_b") or 0.0),
        "preservation_b_in_a": float(raw.get("preservation_b_in_a") or 0.0),
        "conflict": float(raw.get("conflict") or 0.0),
        "contract_version": str(raw.get("contract_version") or CONTRACT_VERSION),
        "raw_model": str(raw.get("raw_model") or ""),
    })


def _json_result(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return {"success": False, "error": raw[:500]}


def _result_ok(value: Any) -> bool:
    return isinstance(value, dict) and value.get("success") is True
