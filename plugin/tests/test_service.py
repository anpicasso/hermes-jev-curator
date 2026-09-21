"""Service facade tests: dry-by-default runs, claim refusal, bounded records, safe doctor.

The engine is stubbed throughout, so these tests pin CuratorService's own contract:
observe never mutates, explicit apply is refused outside ``mode=apply``, a busy profile
claim refuses before the engine is touched, run persists a bounded report/state summary
with no skill bodies, status tolerates absent/corrupt state, doctor never echoes
credentials or full credential-bearing URLs, and containment handlers keep the public
methods returning mappings.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest import mock

from plugin import inventory, state
from plugin.engine import CuratorEngine
from plugin.models import MergePlan, RelationJudgment, Settings, SkillArtifact
from plugin.service import CuratorService, build_service


CONTRACT_VERSION = "skill-relations-v1"
BODY_MARKER = "SKILL-BODY-MARKER"
SKILL_BODY = BODY_MARKER + "-" + "x" * 100_000
RUN_ID_RE = r"^\d{8}T\d{6}Z-\d{6}$"


def empty_scan(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True, "mode": "observe", "contract_version": CONTRACT_VERSION,
        "inventory": [], "candidates": [], "baseline": {}, "judgments": [],
        "skipped": [], "errors": [], "latency_ms": 1,
    }
    payload.update(overrides)
    return payload


def make_artifact(name: str, *, use_count: int = 0) -> SkillArtifact:
    return SkillArtifact(name=name, path=Path("/tmp") / name, description=f"{name} skill",
                         text=f"body of {name}", digest=f"digest-{name}", use_count=use_count)


def make_judgment(a: str = "alpha", b: str = "beta", **overrides: Any) -> RelationJudgment:
    base = RelationJudgment(
        a=a, b=b, a_digest=f"digest-{a}", b_digest=f"digest-{b}",
        relation="duplicate", confidence=0.95,
        probabilities={"duplicate": 0.95, "unrelated": 0.05},
        coverage=0.9, preservation_a_in_b=0.95, preservation_b_in_a=0.95, conflict=0.0,
        contract_version=CONTRACT_VERSION, raw_model="jev-latest",
    )
    return replace(base, **overrides) if overrides else base


def make_plan(plan_id: str = "merge-abc123", canonical: str = "alpha",
              absorbed: tuple[str, ...] = ("beta",)) -> MergePlan:
    return MergePlan(
        plan_id=plan_id, canonical=canonical, canonical_digest=f"digest-{canonical}",
        absorbed=absorbed, absorbed_digests={name: f"digest-{name}" for name in absorbed},
        relation_keys=tuple("::".join(sorted((canonical, name))) for name in absorbed),
        status="validated", metadata={"graph_version": "skill-graph-v1"},
    )


class StubEngine:
    """Canned engine: records every call, returns copies, raises only for ``raise_on``."""

    def __init__(self, *, scan=None, plans=None, execution=None, status=None, review=None,
                 raise_on=(), error=None):
        self.calls: list[tuple] = []
        self.scan_result = empty_scan() if scan is None else scan
        self.plans = [] if plans is None else list(plans)
        self.execution = {"ok": True, "applied": []} if execution is None else execution
        self.status_result = {
            "ok": True, "mode": "observe", "managed_skills": 0, "protected_skills": 0,
            "provider": "typesafe", "network_enabled": True,
        } if status is None else status
        self.review_result = {"ok": True, "judgment": {}} if review is None else review
        self.raise_on = set(raise_on)
        self.error = error or RuntimeError("stub engine failure")

    def _maybe_raise(self, name: str) -> None:
        if name in self.raise_on:
            raise self.error

    def status(self):
        self.calls.append(("status",))
        self._maybe_raise("status")
        return dict(self.status_result)

    def scan(self, names=None, *, use_jev=True):
        self.calls.append(("scan", use_jev))
        self._maybe_raise("scan")
        return dict(self.scan_result)

    def review(self, name):
        self.calls.append(("review", name))
        self._maybe_raise("review")
        return dict(self.review_result)

    def build_plans(self, scan, *, max_plans=5):
        self.calls.append(("build_plans", max_plans))
        self._maybe_raise("build_plans")
        return list(self.plans)

    def apply(self, plans):
        self.calls.append(("apply", tuple(plan.plan_id for plan in plans)))
        self._maybe_raise("apply")
        return dict(self.execution)


class ServiceTestCase(unittest.TestCase):
    ENV_KEYS = ("HERMES_HOME", "JEV_CURATOR_AUDIT_MAX_BYTES")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        saved = {key: os.environ.get(key) for key in self.ENV_KEYS}
        self.addCleanup(self._restore_env, saved)
        os.environ["HERMES_HOME"] = str(self.home / "hermes")
        os.environ.pop("JEV_CURATOR_AUDIT_MAX_BYTES", None)

    def _restore_env(self, saved):
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def make_service(self, *, mode: str = "observe", engine=None, **settings: Any) -> CuratorService:
        service = CuratorService(Settings(mode=mode, **settings))
        service.engine = engine if engine is not None else StubEngine()
        return service

    def payload(self) -> dict[str, Any]:
        return empty_scan(
            inventory=[{"name": "alpha", "text": SKILL_BODY, "digest": "digest-alpha"}],
            candidates=[{"a": "alpha", "b": "beta", "a_digest": "digest-alpha",
                         "b_digest": "digest-beta", "similarity": 1.0, "signals": ["explicit"]}],
            judgments=[asdict(make_judgment())],
        )

    def check(self, result: Mapping[str, Any], name: str) -> dict[str, Any]:
        rows = [row for row in result["checks"] if row["name"] == name]
        self.assertEqual(len(rows), 1, f"expected exactly one {name!r} check, got {result['checks']}")
        return rows[0]

    def call_names(self, engine: StubEngine) -> list[str]:
        return [call[0] for call in engine.calls]

    def audit_events(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in
                state.audit_log_path().read_text(encoding="utf-8").splitlines() if line.strip()]


class RunDryTests(ServiceTestCase):
    def test_observe_run_is_dry_and_records_bounded_artifacts(self):
        payload = self.payload()
        payload["skipped"] = [{"pair": "gamma::omega", "reason": "request-budget",
                               "requests": 3}]
        engine = StubEngine(scan=payload, plans=[make_plan()])
        service = self.make_service(mode="observe", engine=engine)

        result = service.run()

        self.assertIsInstance(result, Mapping)
        self.assertTrue(result["ok"])
        self.assertFalse(result["apply_requested"])
        self.assertEqual(self.call_names(engine), ["scan", "build_plans"])
        self.assertEqual(result["execution"],
                         {"ok": True, "applied": [], "message": "observe-only; no skill mutations"})
        self.assertEqual((result["inventory_count"], result["candidate_count"], result["judgment_count"]),
                         (1, 1, 1))
        self.assertEqual(result["skipped_count"], 1)
        self.assertEqual(result["skipped"], payload["skipped"])

        summary = state.load_state()["last_run"]
        self.assertEqual(set(summary), {"run_id", "ok", "mode", "finished_at", "inventory_count",
                                        "candidate_count", "judgment_count", "skipped_count",
                                        "applied_count"})
        self.assertEqual(summary["mode"], "observe")
        self.assertEqual(summary["applied_count"], 0)
        self.assertEqual(summary["finished_at"], result["finished_at"])
        self.assertRegex(summary["run_id"], RUN_ID_RE)
        for value in summary.values():
            self.assertNotIsInstance(value, (dict, list), "state summary must stay scalar")

        report_path = Path(result["report"])
        self.assertEqual(report_path.parent, state.reports_dir())
        self.assertEqual(report_path.stem, summary["run_id"])
        raw_report = report_path.read_text(encoding="utf-8")
        report = json.loads(raw_report)
        self.assertEqual(report["mode"], "observe")
        self.assertEqual(report["plans"][0]["plan_id"], "merge-abc123")
        self.assertEqual(report["plans"][0]["absorbed"], ["beta"])
        self.assertEqual(report["execution"]["applied"], [])
        self.assertEqual(report["skipped"], payload["skipped"])
        self.assertLess(len(raw_report), 20_000)

        self.assertNotIn(SKILL_BODY, raw_report)
        self.assertNotIn(BODY_MARKER, raw_report)
        self.assertNotIn(SKILL_BODY, json.dumps(result))
        self.assertNotIn(BODY_MARKER, state.state_path().read_text(encoding="utf-8"))
        self.assertNotIn(BODY_MARKER, state.audit_log_path().read_text(encoding="utf-8"))
        self.assertIn("run", [row["event"] for row in self.audit_events()])
        self.assertEqual(service.status()["last_run"]["run_id"], summary["run_id"])

    def test_armed_mode_without_the_flag_still_runs_dry(self):
        engine = StubEngine(scan=self.payload(), plans=[make_plan()])
        service = self.make_service(mode="apply", engine=engine)

        result = service.run()

        self.assertTrue(result["ok"])
        self.assertFalse(result["apply_requested"])
        self.assertEqual(self.call_names(engine), ["scan", "build_plans"])
        self.assertEqual(result["execution"]["message"], "observe-only; no skill mutations")
        self.assertEqual(state.load_state()["last_run"]["applied_count"], 0)

    def test_explicit_apply_is_refused_unless_mode_is_apply(self):
        for mode in ("observe", "guard", "off"):
            with self.subTest(mode=mode):
                engine = StubEngine(scan=self.payload(), plans=[make_plan()])
                service = self.make_service(mode=mode, engine=engine)

                result = service.run(apply=True)

                self.assertIsInstance(result, Mapping)
                self.assertFalse(result["ok"])
                self.assertIn("apply requires mode=apply", result["error"])
                self.assertEqual(engine.calls, [], "a refused apply must not touch the engine")
                self.assertEqual(state.load_state(), {})
                self.assertEqual(state.list_reports(), [])

    def test_apply_mode_with_flag_calls_engine_apply_exactly_once(self):
        execution = {"ok": True, "snapshot": "snap-1",
                     "applied": [{"plan_id": "merge-abc123", "skill": "beta",
                                  "result": {"success": True}}]}
        engine = StubEngine(scan=self.payload(), plans=[make_plan()], execution=execution)
        service = self.make_service(mode="apply", engine=engine)

        result = service.run(apply=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["apply_requested"])
        self.assertEqual([call for call in engine.calls if call[0] == "apply"],
                         [("apply", ("merge-abc123",))])
        self.assertEqual(result["execution"], execution)
        self.assertEqual(state.load_state()["last_run"]["applied_count"], 1)
        report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
        self.assertEqual(report["execution"]["applied"][0]["skill"], "beta")

    def test_guard_run_installs_only_current_hash_bound_authority(self):
        from plugin import guard

        engine = StubEngine(scan=self.payload(), plans=[make_plan()])
        service = self.make_service(mode="guard", engine=engine)
        result = service.run()

        self.assertTrue(result["authorization"]["ok"])
        self.assertEqual(result["authorization"]["installed"], ["merge-abc123"])
        self.assertEqual(result["authorization"]["dropped"], [])
        self.assertEqual([row["plan_id"] for row in guard.load_authorized_plans()],
                         ["merge-abc123"])

    def test_guard_run_reports_authority_removed_by_replacement(self):
        from plugin import guard

        old = make_plan("merge-old", "legacy", ("stale",))
        current = make_plan("merge-new", "alpha", ("beta",))
        self.assertEqual(guard.install_authorized_plan(old), "merge-old")
        service = self.make_service(mode="guard", engine=StubEngine(
            scan=self.payload(), plans=[current]))

        result = service.run()

        self.assertEqual(result["authorization"], {
            "installed": ["merge-new"], "dropped": ["merge-old"], "ok": True,
        })
        self.assertEqual([row["plan_id"] for row in guard.load_authorized_plans()],
                         ["merge-new"])


class ClaimTests(ServiceTestCase):
    def test_busy_profile_claim_refuses_before_touching_the_engine(self):
        engine = StubEngine(scan=self.payload(), plans=[make_plan()])
        service = self.make_service(mode="observe", engine=engine)

        with state.claim_lock() as held:
            self.assertTrue(held)
            result = service.run()

        self.assertIsInstance(result, Mapping)
        self.assertFalse(result["ok"])
        self.assertIn("claim", result["error"])
        self.assertEqual(engine.calls, [])
        self.assertEqual(state.load_state(), {})
        self.assertEqual(state.list_reports(), [])

    def test_claim_is_sized_to_the_run_budget(self):
        recorded: dict[str, float] = {}

        @contextlib.contextmanager
        def busy(*, stale_seconds: float = 0.0):
            recorded["stale_seconds"] = stale_seconds
            yield False

        for settings, expected in (
            ({"timeout_seconds": 25.0, "max_requests": 50}, 25.0 * 50 + 60.0),
            ({"timeout_seconds": 1.0, "max_requests": 1}, 300.0),
        ):
            with self.subTest(expected=expected):
                engine = StubEngine()
                service = self.make_service(mode="observe", engine=engine, **settings)
                with mock.patch.object(state, "claim_lock", busy):
                    result = service.run()
                self.assertFalse(result["ok"])
                self.assertEqual(recorded["stale_seconds"], expected)
                self.assertEqual(engine.calls, [])


class StatusTests(ServiceTestCase):
    def test_status_tolerates_absent_and_hostile_state(self):
        engine = StubEngine(status={"ok": True, "mode": "observe", "managed_skills": 3,
                                    "protected_skills": 1, "provider": "typesafe",
                                    "network_enabled": True})
        service = self.make_service(engine=engine)

        first = service.status()
        self.assertIsInstance(first, Mapping)
        self.assertEqual(first["managed_skills"], 3)
        self.assertIsNone(first["last_run"])

        state.state_path().parent.mkdir(parents=True, exist_ok=True)
        state.state_path().write_text("{not json", encoding="utf-8")
        self.assertIsNone(service.status()["last_run"])

        state.write_json(state.state_path(), ["not", "a", "mapping"])
        self.assertIsNone(service.status()["last_run"])

        state.save_state({"last_run": "junk"})
        self.assertIsNone(service.status()["last_run"])

        state.save_state({"last_run": {"run_id": "r-1", "ok": True}})
        self.assertEqual(service.status()["last_run"], {"run_id": "r-1", "ok": True})


class DoctorTests(ServiceTestCase):
    def test_doctor_reports_host_only_and_never_echoes_settings(self):
        service = self.make_service(mode="observe", provider="custom",
                                    base_url="https://api.example.com/v1/TOKEN-abc123",
                                    key_env="MY_JEV_KEY_ENV")

        result = service.doctor()

        blob = json.dumps(result)
        self.assertTrue(result["ok"])
        self.assertEqual(self.check(result, "route")["host"], "api.example.com")
        self.assertNotIn("TOKEN-abc123", blob)
        self.assertNotIn("https://api.example.com/v1", blob)
        self.assertNotIn("MY_JEV_KEY_ENV", blob)
        self.assertNotIn("key_env", blob)

    def test_doctor_rejects_credential_bearing_urls_without_echoing_them(self):
        cases = (
            ("https://user:HUNTER2@api.example.com", "HUNTER2"),
            ("https://api.example.com/v1?token=QUERYSECRET123", "QUERYSECRET123"),
            ("https://api.example.com/v1#FRAGSECRET123", "FRAGSECRET123"),
        )
        for base_url, leak in cases:
            with self.subTest(base_url=base_url):
                service = self.make_service(provider="custom", base_url=base_url)
                result = service.doctor()
                blob = json.dumps(result)
                route = self.check(result, "route")
                self.assertFalse(route["ok"])
                self.assertIn("ValueError", route["error"])
                self.assertFalse(result["ok"])
                self.assertNotIn(leak, blob)
                self.assertNotIn(base_url, blob)

        service = self.make_service(provider="typesafe", key_env="MY_JEV_KEY_ENV")
        result = service.doctor()
        blob = json.dumps(result)
        route = self.check(result, "route")
        self.assertFalse(route["ok"])
        self.assertIn("ValueError", route["error"])
        self.assertNotIn("key_env", route["error"], "raw route exception text must not be echoed")
        self.assertNotIn("MY_JEV_KEY_ENV", blob)

    def test_doctor_suppresses_raw_route_exception_text(self):
        service = self.make_service(provider="custom", base_url="https://api.example.com")

        with mock.patch("plugin.service.resolve_route",
                        side_effect=ValueError("https://user:HUNTER2@api.example.com/v1 unusable")):
            result = service.doctor()

        blob = json.dumps(result)
        self.assertFalse(result["ok"])
        route = self.check(result, "route")
        self.assertFalse(route["ok"])
        self.assertIn("ValueError", route["error"])
        for leak in ("HUNTER2", "user:", "api.example.com"):
            self.assertNotIn(leak, blob)

    def test_doctor_flags_apply_mode_as_non_default(self):
        observe = self.make_service(mode="observe").doctor()
        self.assertTrue(self.check(observe, "mutation-default")["ok"])

        armed = self.make_service(mode="apply").doctor()
        self.assertFalse(armed["ok"])
        check = self.check(armed, "mutation-default")
        self.assertFalse(check["ok"])
        self.assertEqual(check["mode"], "apply")
        self.assertIn("--apply", check["note"])


class ContainmentTests(ServiceTestCase):
    def test_doctor_contains_route_and_engine_failures(self):
        service = self.make_service(provider="custom", engine=StubEngine(raise_on={"status"}))

        result = service.doctor()

        self.assertIsInstance(result, Mapping)
        self.assertFalse(result["ok"])
        route = self.check(result, "route")
        self.assertFalse(route["ok"])
        self.assertIn("ValueError", route["error"])
        inventory_check = self.check(result, "inventory")
        self.assertFalse(inventory_check["ok"])
        self.assertIn("RuntimeError", inventory_check["error"])

    def test_doctor_suppresses_raw_inventory_exception_text(self):
        leak = "https://user:HUNTER2@api.example.com/private"
        engine = StubEngine()
        engine.status = mock.Mock(side_effect=RuntimeError(leak))
        service = self.make_service(engine=engine)

        result = service.doctor()

        self.assertFalse(result["ok"])
        self.assertNotIn("HUNTER2", json.dumps(result))
        self.assertNotIn(leak, json.dumps(result))

    def test_run_contains_report_and_state_write_failures(self):
        engine = StubEngine(scan=self.payload(), plans=[make_plan()])
        service = self.make_service(mode="observe", engine=engine)

        with mock.patch.object(state, "write_report", side_effect=RuntimeError("disk full")), \
                mock.patch.object(state, "save_state", side_effect=OSError("read-only home")):
            result = service.run()

        self.assertIsInstance(result, Mapping)
        self.assertTrue(result["ok"])
        self.assertNotIn("report", result)
        self.assertIn("report write failed: RuntimeError", result["warnings"])
        self.assertIn("state write failed: OSError", result["warnings"])

    def test_public_methods_still_return_mappings_when_audit_fails(self):
        engine = StubEngine(scan=self.payload(), plans=[make_plan()])
        service = self.make_service(mode="observe", engine=engine)

        with mock.patch.object(state, "audit", side_effect=RuntimeError("audit down")), \
                mock.patch.object(inventory, "collect_inventory", return_value=[]):
            results = [service.status(), service.scan(), service.review(" alpha "),
                       service.graph(), service.plan(), service.run(), service.doctor()]
            self.assertIsNone(service.lifecycle(skill="alpha", event="deleted"))

        for result in results:
            self.assertIsInstance(result, Mapping)
        self.assertIn(("review", "alpha"), engine.calls)


class GraphAndPlanTests(ServiceTestCase):
    def test_scan_use_jev_flag_follows_mode(self):
        for mode, expected in (("observe", True), ("guard", True), ("off", False)):
            with self.subTest(mode=mode):
                engine = StubEngine(scan=empty_scan())
                service = self.make_service(mode=mode, engine=engine)
                service.scan()
                self.assertEqual(engine.calls, [("scan", expected)])

    def test_graph_and_plan_are_mapping_summaries(self):
        engine = StubEngine(scan=empty_scan(judgments=[asdict(make_judgment())]),
                            plans=[make_plan(), make_plan("merge-def456", "gamma", ("delta",))])
        service = self.make_service(mode="observe", engine=engine)

        with mock.patch.object(inventory, "collect_inventory",
                               return_value=[make_artifact("alpha", use_count=5), make_artifact("beta")]):
            graph = service.graph()
            plan = service.plan()

        self.assertIsInstance(graph, Mapping)
        self.assertTrue(graph["ok"])
        self.assertEqual(graph["version"], "skill-graph-v1")
        self.assertEqual(graph["nodes"], ["alpha", "beta"])
        self.assertEqual(graph["authorized_edges"], 1)
        self.assertEqual(graph["refusals"], {})
        self.assertEqual([edge["canonical"] for edge in graph["edges"]], ["alpha"])

        self.assertIsInstance(plan, Mapping)
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["mode"], "observe")
        self.assertEqual(plan["applicable"], 2)
        self.assertEqual([row["plan_id"] for row in plan["plans"]],
                         ["merge-abc123", "merge-def456"])
        self.assertEqual(plan["errors"], [])

    def test_review_strips_the_skill_name(self):
        engine = StubEngine(review={"ok": True, "judgment": {"relation": "unrelated"}})
        service = self.make_service(engine=engine)

        result = service.review("  alpha  ")

        self.assertEqual(result, {"ok": True, "judgment": {"relation": "unrelated"}})
        self.assertIn(("review", "alpha"), engine.calls)


class SettingsSeamTests(ServiceTestCase):
    def test_build_service_reads_ctx_config_and_sanitizes_it(self):
        class Ctx:
            def get_config(self, key, default=None):
                return {"mode": " APPLY ", "provider": "OpenRouter", "timeout_seconds": 9999,
                        "max_requests": 0, "jev_model": "jev-latest"}.get(key, default)

        service = build_service(Ctx())

        self.assertIsInstance(service, CuratorService)
        self.assertIsInstance(service.engine, CuratorEngine)
        self.assertEqual(service.settings.mode, "apply")
        self.assertEqual(service.settings.provider, "openrouter")
        self.assertEqual(service.settings.timeout_seconds, 120.0)
        self.assertEqual(service.settings.max_requests, 1)
        self.assertEqual(service.settings.model, "jev-latest")


if __name__ == "__main__":
    unittest.main()
