"""Engine integration tests: the eight safety behaviors, with every host seam stubbed.

Scope (one class per behavior):
  1. ObserveModeTests   -- observe mode dispatches nothing and takes no snapshot.
  2. RequestBudgetTests -- scan issues at most ``max_requests`` Jev calls, on the
                           top-ranked pairs; ``off``/zero budget issue none.
  3. JevFailureTests    -- a failing Jev call is an error, never a judgment.
  4. TruncationTests    -- explicitly truncated state skips Jev, yields
                           ``insufficient_evidence``, and builds no applicable plan.
  5. ApplyModeGateTests -- apply refuses unless mode == "apply" and a ctx exists.
  6. StaleHashTests     -- changed/missing content digests refuse before backup.
  7. BackupOrderTests   -- the snapshot is taken before the first dispatch; a failed
                           or unavailable snapshot refuses with zero dispatches.
  8. WritePathTests     -- every mutation is ``ctx.dispatch_tool("skill_manage", ...)``
                           under the background_review write origin, restored after.
  9. ApplyGateTests     -- chained/overlapping plans and an enabled (or unverifiable)
                           ``skills.write_approval`` refuse before the backup.

Everything is in-process and hermetic: the Jev transport, skill inventory, relation
cache, ``agent.curator_backup``, ``tools.write_approval``, and
``tools.skill_provenance`` are replaced with recording stubs; the plugin context is a
stub returning the JSON-string shape of the real tool registry and raising if the
engine dispatches any tool these tests do not expect. No network, no credentials, no
real ``~/.hermes`` access.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

from plugin import engine as engine_module
from plugin import state as state_module
from plugin.candidates import generate_candidates
from plugin.models import MergePlan, Settings, SkillArtifact
from plugin.transport import JevResponse

CONTRACT_VERSION = "skill-relations-v1"


# --- fixtures -------------------------------------------------------------------------

def make_artifact(name: str, *, text: str | None = None, digest: str | None = None,
                  **overrides: Any) -> SkillArtifact:
    """Similar-body artifact so lexical candidates form between any two of them."""
    return SkillArtifact(
        name=name,
        path=Path("/skills") / name,
        description="review pull request diffs safely",
        text=text if text is not None else (
            f"review pull request diffs and branch history safely for {name} repos. "
            "shared tokens repeated for lexical overlap. " * 3),
        digest=digest if digest is not None else f"digest-{name}",
        **overrides,
    )


def default_inventory() -> list[SkillArtifact]:
    return [make_artifact(name) for name in ("alpha", "beta", "gamma")]


def canned_response(relation: str = "duplicate", confidence: float = 0.95,
                    coverage: float = 0.9, a_in_b: float = 0.95, b_in_a: float = 0.95,
                    conflict: float = 0.05, model: str = "jev-test") -> JevResponse:
    """A fully shaped JevResponse; transport validation is stubbed out of this suite."""
    return JevResponse(
        answers={
            "relation": {"choice": relation, "confidence": confidence,
                         "probabilities": {relation: confidence}},
            "coverage": {"noul": coverage},
            "a_in_b": {"noul": a_in_b},
            "b_in_a": {"noul": b_in_a},
            "conflict": {"noul": conflict},
        },
        model=model, provider="typesafe", usage={}, attempts=1, http_status=200,
        request_id="req-1",
    )


def make_plan(canonical: str = "alpha", absorbed: tuple[str, ...] = ("beta",), *,
              status: str = "validated", blockers: tuple[str, ...] = (),
              plan_id: str = "merge-abc123") -> MergePlan:
    return MergePlan(
        plan_id=plan_id,
        canonical=canonical,
        canonical_digest=f"digest-{canonical}",
        absorbed=tuple(absorbed),
        absorbed_digests={name: f"digest-{name}" for name in absorbed},
        relation_keys=tuple("::".join(sorted((canonical, name))) for name in absorbed),
        status=status,
        blockers=tuple(blockers),
    )


def make_provenance_module() -> types.ModuleType:
    """Mirror of ``tools.skill_provenance``: contextvars + token set/reset contract."""
    module = types.ModuleType("tools.skill_provenance")
    origin = contextvars.ContextVar("skill_write_origin", default="foreground")
    attended = contextvars.ContextVar("review_attended", default=False)
    module.BACKGROUND_REVIEW = "background_review"
    module.set_current_write_origin = origin.set
    module.reset_current_write_origin = origin.reset
    module.get_current_write_origin = origin.get
    module.set_review_attended = lambda value: attended.set(bool(value))
    module.reset_review_attended = attended.reset
    module.get_review_attended = attended.get
    return module


class StubContext:
    """Recording plugin context; returns the JSON-string shape of the real registry."""

    def __init__(self, provenance: types.ModuleType, events: list[tuple],
                 *, view_failures: tuple[str, ...] = (), manage_results: tuple[Any, ...] = (),
                 on_archive: Any = None):
        self.provenance = provenance
        self.events = events
        self.view_failures = set(view_failures)
        self.manage_results = list(manage_results)
        self.on_archive = on_archive
        self.calls: list[dict[str, Any]] = []

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs: Any) -> str:
        record = {
            "tool": tool_name,
            "args": dict(args),
            "origin": self.provenance.get_current_write_origin(),
            "attended": self.provenance.get_review_attended(),
        }
        self.calls.append(record)
        self.events.append(("dispatch", tool_name, record["args"].get("name"), record["origin"]))
        if tool_name == "skill_view":
            name = str(record["args"].get("name") or "")
            if name in self.view_failures:
                return json.dumps({"success": False, "error": f"unknown skill: {name}"})
            return json.dumps({"success": True, "name": name})
        if tool_name == "skill_manage":
            raw = (self.manage_results.pop(0) if self.manage_results else json.dumps({
                "success": True, "_archived": True,
                "message": f"Skill '{record['args'].get('name')}' archived."}))
            decoded = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(decoded, dict) and decoded.get("_archived") is True and self.on_archive:
                self.on_archive(str(record["args"].get("name") or ""))
            return raw
        raise AssertionError(f"engine dispatched unexpected tool {tool_name!r}")


class EngineHarness:
    """Installs every stubbed host seam the engine touches; yields the engine + spies."""

    def __init__(self, *, mode: str = "observe", inventory: list[SkillArtifact] | None = None,
                 request_impl: Any = None, snapshot: Path | None = Path("/backups/snap-1"),
                 snapshot_error: Exception | None = None, approval_enabled: bool = False,
                 approval_module: bool = True, manage_results: tuple[Any, ...] = (),
                 view_failures: tuple[str, ...] = (), auto_archive: bool = True,
                 post_archive_digests: dict[str, str] | None = None, **settings: Any):
        settings.setdefault("allow_content_egress", True)
        self.settings = Settings(mode=mode, **settings)
        self.inventory = list(inventory if inventory is not None else default_inventory())
        self.request_calls: list[set[str]] = []
        self.request_impl = request_impl
        self.snapshot = snapshot
        self.snapshot_error = snapshot_error
        self.snapshot_calls: list[dict[str, Any]] = []
        self.approval_enabled = approval_enabled
        self.approval_module = approval_module
        self.auto_archive = auto_archive
        self.post_archive_digests = dict(post_archive_digests or {})
        self.discarded_pending: list[tuple[str, str]] = []
        self.events: list[tuple] = []
        self.provenance = make_provenance_module()
        self.ctx = StubContext(self.provenance, self.events,
                               view_failures=view_failures, manage_results=manage_results,
                               on_archive=self._archive)
        self.inventory_calls: list[bool] = []
        self.cache_saves = 0
        self.engine: engine_module.CuratorEngine | None = None
        self._stack: contextlib.ExitStack | None = None

    # -- stubs -------------------------------------------------------------------
    def _collect_inventory(self, *, skills_root: Path | None = None,
                           include_unmanaged: bool = False,
                           names: Any = None) -> list[SkillArtifact]:
        self.inventory_calls.append(include_unmanaged)
        wanted = {str(name) for name in names} if names is not None else None
        return [item for item in self.inventory if wanted is None or item.name in wanted]

    def _archive(self, name: str) -> None:
        if not self.auto_archive:
            return
        self.inventory = [item for item in self.inventory if item.name != name]
        if self.post_archive_digests:
            self.inventory = [replace(item, digest=self.post_archive_digests[item.name])
                              if item.name in self.post_archive_digests else item
                              for item in self.inventory]

    def _request(self, state: dict[str, Any], questions: Any, settings: Settings) -> JevResponse:
        self.request_calls.append({state.get("skill_a_name"), state.get("skill_b_name")})
        if self.request_impl is not None:
            return self.request_impl(state, questions, settings)
        return canned_response()

    def _snapshot_skills(self, reason: str = "manual", *, protect_ids: Any = None) -> Path | None:
        self.snapshot_calls.append({"reason": reason, "protect_ids": protect_ids})
        self.events.append(("snapshot", reason))
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return self.snapshot

    def _save_relation_cache(self, cache: Any, **kwargs: Any) -> Path:
        self.cache_saves += 1
        return Path("/state/relations.json")

    # -- lifecycle ----------------------------------------------------------------
    def __enter__(self) -> "EngineHarness":
        stack = contextlib.ExitStack()
        self._stack = stack
        stack.enter_context(mock.patch.object(engine_module, "collect_inventory",
                                              self._collect_inventory))
        stack.enter_context(mock.patch.object(engine_module, "request", self._request))
        stack.enter_context(mock.patch.object(state_module, "load_relation_cache", lambda: {}))
        stack.enter_context(mock.patch.object(state_module, "cached_relation", lambda *a, **k: None))
        stack.enter_context(mock.patch.object(state_module, "remember_relation", lambda *a, **k: "key"))
        stack.enter_context(mock.patch.object(state_module, "save_relation_cache",
                                              self._save_relation_cache))
        backup = types.ModuleType("agent.curator_backup")
        backup.snapshot_skills = self._snapshot_skills
        approval = types.ModuleType("tools.write_approval")
        approval.SKILLS = "skills"
        if self.approval_module:
            approval.write_approval_enabled = lambda subsystem: bool(self.approval_enabled)
            def discard_pending(subsystem, pending_id):
                self.discarded_pending.append((subsystem, pending_id))
                return True
            approval.discard_pending = discard_pending
        stack.enter_context(mock.patch.dict(sys.modules, {
            "tools.skill_provenance": self.provenance,
            "agent.curator_backup": backup,
            "tools.write_approval": approval,
        }))
        self.engine = engine_module.CuratorEngine(self.settings, self.ctx)
        return self

    def __exit__(self, *exc: Any) -> None:
        assert self._stack is not None
        self._stack.close()


# --- 1. observe mode ------------------------------------------------------------------

class ObserveModeTests(unittest.TestCase):
    def test_observe_scan_status_and_review_dispatch_nothing(self):
        with EngineHarness(mode="observe") as h:
            status = h.engine.status()
            scan = h.engine.scan()
            review = h.engine.review("alpha")
            pair_review = h.engine.review("alpha", "beta")

        self.assertEqual(h.ctx.calls, [], "observe mode must not dispatch any tool call")
        self.assertEqual(h.snapshot_calls, [], "observe mode must not take a backup")
        self.assertTrue(status["ok"])
        self.assertEqual(status["mode"], "observe")
        self.assertTrue(scan["ok"])
        self.assertEqual(scan["mode"], "observe")
        self.assertEqual(len(scan["judgments"]), len(scan["candidates"]))
        self.assertGreater(len(scan["judgments"]), 0)
        self.assertTrue(review["ok"])
        self.assertTrue(pair_review["ok"])
        self.assertEqual(h.cache_saves, 2, "the relation cache is the only persisted effect")


# --- 2. bounded scan / request budget -------------------------------------------------

class RequestBudgetTests(unittest.TestCase):
    def test_content_egress_requires_explicit_consent(self):
        with EngineHarness(mode="observe", allow_content_egress=False) as h:
            assert h.engine is not None
            status = h.engine.status()
            scan = h.engine.scan()

        self.assertFalse(status["network_enabled"])
        self.assertFalse(scan["content_egress_enabled"])
        self.assertEqual(scan["jev_skipped"], "content egress is disabled")
        self.assertEqual(h.request_calls, [])
        self.assertEqual(scan["judgments"], [])

    def test_request_budget_bounds_jev_calls_to_the_top_ranked_pairs(self):
        skills = [make_artifact(name) for name in
                  ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")]
        with EngineHarness(mode="observe", inventory=skills, max_requests=2) as h:
            scan = h.engine.scan()

        expected = generate_candidates(skills, top_k=h.settings.top_k,
                                       max_pairs=h.settings.max_pairs)
        self.assertGreater(len(expected), 2, "fixture must produce more pairs than the budget")
        self.assertEqual([row["a"] for row in scan["candidates"]],
                         [pair.a for pair in expected])
        self.assertEqual(len(h.request_calls), 2, "scan must not exceed the request budget")
        self.assertEqual(len(scan["judgments"]), 2)
        self.assertEqual({frozenset(names) for names in h.request_calls},
                         {frozenset((pair.a, pair.b)) for pair in expected[:2]},
                         "the budget must cover the top-ranked pairs, not an arbitrary subset")

    def test_zero_request_budget_issues_no_jev_calls(self):
        with EngineHarness(mode="observe", max_requests=0) as h:
            scan = h.engine.scan()

        self.assertEqual(h.request_calls, [])
        self.assertEqual(scan["judgments"], [])
        self.assertGreater(len(scan["candidates"]), 0)

    def test_off_mode_issues_no_jev_calls(self):
        with EngineHarness(mode="off") as h:
            scan = h.engine.scan()
            review = h.engine.review("alpha", "beta")

        self.assertEqual(h.request_calls, [])
        self.assertEqual(scan["judgments"], [])
        self.assertFalse(review["ok"])
        self.assertIn("mode is off", review["error"])
        self.assertEqual(h.ctx.calls, [])


# --- 3. Jev failure is an error, never an approval -------------------------------------

class JevFailureTests(unittest.TestCase):
    @staticmethod
    def _fail(state: dict[str, Any], questions: Any, settings: Settings) -> JevResponse:
        raise RuntimeError("Jev request failed after bounded retries: RuntimeError")

    def test_every_failed_request_is_an_error_with_no_judgment(self):
        with EngineHarness(mode="observe", request_impl=self._fail) as h:
            scan = h.engine.scan()

        self.assertFalse(scan["ok"])
        self.assertEqual(scan["judgments"], [], "a failure must not fabricate a judgment")
        self.assertGreater(len(scan["candidates"]), 0)
        self.assertEqual(len(scan["errors"]), len(scan["candidates"]))
        for row in scan["errors"]:
            self.assertIn("RuntimeError", row["error"])
            self.assertTrue(row["pair"])

    def test_partial_failure_still_reports_not_ok(self):
        def fail_beta_gamma(state: dict[str, Any], questions: Any, settings: Settings) -> JevResponse:
            if {state["skill_a_name"], state["skill_b_name"]} == {"beta", "gamma"}:
                raise RuntimeError("boom")
            return canned_response()

        with EngineHarness(mode="observe", request_impl=fail_beta_gamma) as h:
            scan = h.engine.scan()

        self.assertFalse(scan["ok"], "one failed pair must poison the scan verdict")
        self.assertEqual(len(scan["errors"]), 1)
        self.assertEqual(scan["errors"][0]["pair"], "beta::gamma")
        judged_keys = {f"{row['a']}::{row['b']}" for row in scan["judgments"]}
        self.assertEqual(judged_keys, {"alpha::beta", "alpha::gamma"})
        self.assertNotIn("beta::gamma", judged_keys)

    def test_pair_review_failure_returns_an_error_not_a_judgment(self):
        with EngineHarness(mode="observe", request_impl=self._fail) as h:
            result = h.engine.review("alpha", "beta")

        self.assertFalse(result["ok"])
        self.assertNotIn("judgment", result)
        self.assertIn("RuntimeError", result["error"])

    def test_unresolvable_provider_is_an_error_result_not_an_exception(self):
        """A route/config failure is a Jev-side failure like any other.

        ``resolve_route`` raises ValueError for an unknown provider, and that value comes
        straight from operator config (e.g. a typo, or the unset-setting path where every
        ``ctx.get_config(key, None)`` is ``None``). Every other Jev failure is collected
        into ``scan["errors"]``; an escaping exception instead crashes
        ``CuratorService.run`` and takes the whole plugin down on a config typo.
        """
        with EngineHarness(mode="observe", provider="not-a-provider") as h:
            try:
                scan = h.engine.scan()
            except Exception as exc:  # noqa: BLE001 -- the failure under test
                self.fail("scan must report an unresolvable provider as an error result, "
                          f"not raise {type(exc).__name__}: {exc}")

        self.assertFalse(scan["ok"])
        self.assertIn("provider", json.dumps(scan.get("errors", [])),
                      "the route failure must be surfaced in scan errors")


# --- 4. explicit truncation never authorizes merging -----------------------------------

class TruncationTests(unittest.TestCase):
    def test_truncated_pair_skips_jev_and_builds_no_applicable_plan(self):
        skills = [make_artifact("huge-a", text="A" * 8_000),
                  make_artifact("huge-b", text="B" * 8_000)]

        def explode(state: dict[str, Any], questions: Any, settings: Settings) -> JevResponse:
            raise AssertionError("Jev must never be asked about explicitly truncated state")

        with EngineHarness(mode="observe", inventory=skills, max_state_chars=4_000,
                           request_impl=explode) as h:
            scan = h.engine.scan()
            plans = h.engine.build_plans(scan)

        self.assertEqual(h.request_calls, [], "truncated pairs must not reach the network")
        self.assertTrue(scan["ok"])
        self.assertEqual(len(scan["judgments"]), 1)
        judgment = scan["judgments"][0]
        self.assertEqual(judgment["relation"], "insufficient_evidence")
        self.assertEqual(judgment["coverage"], 0.0)
        self.assertEqual(judgment["preservation_a_in_b"], 0.0)
        self.assertEqual(judgment["preservation_b_in_a"], 0.0)
        self.assertEqual(judgment["contract_version"], CONTRACT_VERSION)

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].status, "noop")
        self.assertEqual(plans[0].absorbed, ())
        self.assertFalse(plans[0].applicable, "truncated evidence must never authorize a merge")


# --- 5. apply requires mode=apply ------------------------------------------------------

class ApplyModeGateTests(unittest.TestCase):
    def test_apply_refused_in_every_other_mode(self):
        for mode in ("observe", "guard", "off"):
            with self.subTest(mode=mode):
                with EngineHarness(mode=mode) as h:
                    result = h.engine.apply([make_plan()])
                self.assertFalse(result["ok"])
                self.assertIn("mode", result["error"])
                self.assertIn("apply", result["error"])
                self.assertEqual(h.ctx.calls, [])
                self.assertEqual(h.snapshot_calls, [])

    def test_apply_without_context_is_refused(self):
        engine = engine_module.CuratorEngine(Settings(mode="apply"), None)
        result = engine.apply([make_plan()])
        self.assertFalse(result["ok"])
        self.assertIn("context", result["error"])

    def test_apply_without_an_applicable_plan_is_a_quiet_noop(self):
        blocked = make_plan(status="blocked", blockers=("protected:beta:pinned",), plan_id="merge-1")
        proposed = make_plan(status="proposed", plan_id="merge-2")
        with EngineHarness(mode="apply") as h:
            result = h.engine.apply([blocked, proposed])

        self.assertTrue(result["ok"])
        self.assertEqual(result["applied"], [])
        self.assertEqual(h.ctx.calls, [])
        self.assertEqual(h.snapshot_calls, [])


# --- 6. stale content hashes stop apply ------------------------------------------------

class StaleHashTests(unittest.TestCase):
    def test_protected_target_is_refused_before_backup(self):
        inventory = [make_artifact("alpha"),
                     make_artifact("beta", protected_reasons=("name-path-mismatch",))]
        with EngineHarness(mode="apply", inventory=inventory) as h:
            result = h.engine.apply([make_plan()])
        self.assertFalse(result["ok"])
        self.assertIn("protected", result["error"])
        self.assertEqual(h.snapshot_calls, [])
        self.assertEqual(h.ctx.calls, [])

    def test_changed_or_missing_digests_refuse_before_backup_and_dispatch(self):
        cases = {
            "canonical changed": [make_artifact("alpha", digest="digest-alpha-NEW"),
                                  make_artifact("beta")],
            "absorbed changed": [make_artifact("alpha"),
                                 make_artifact("beta", digest="digest-beta-NEW")],
            "absorbed missing": [make_artifact("alpha")],
        }
        for label, inventory in cases.items():
            with self.subTest(label=label):
                with EngineHarness(mode="apply", inventory=inventory) as h:
                    result = h.engine.apply([make_plan()])
                self.assertFalse(result["ok"])
                self.assertIn("rescan", result["error"])
                self.assertEqual(h.ctx.calls, [], "a stale plan must not reach the host")
                self.assertEqual(h.snapshot_calls, [], "stale refusal must precede the backup")

    def test_plan_that_omits_an_absorbed_digest_is_refused_unverified(self):
        """Apply must refuse any member whose digest is absent from the plan."""
        plan = MergePlan(plan_id="merge-x", canonical="alpha", canonical_digest="digest-alpha",
                         absorbed=("beta",), absorbed_digests={}, relation_keys=("alpha::beta",),
                         status="validated", blockers=())
        with EngineHarness(mode="apply",
                           inventory=[make_artifact("alpha"), make_artifact("beta")]) as h:
            result = h.engine.apply([plan])

        self.assertFalse(result["ok"])
        self.assertTrue(result["error"])
        self.assertEqual(h.ctx.calls, [],
                         "an unverifiable member must not be deleted without a content check")


# --- 7. backup happens before dispatch -------------------------------------------------

class BackupOrderTests(unittest.TestCase):
    def test_snapshot_precedes_every_dispatch(self):
        with EngineHarness(mode="apply", snapshot=Path("/backups/snap-1")) as h:
            result = h.engine.apply([make_plan()])

        self.assertTrue(result["ok"])
        self.assertEqual(result["snapshot"], "snap-1")
        self.assertEqual([row["reason"] for row in h.snapshot_calls],
                         ["pre-jev-curator-apply"])
        self.assertTrue(h.ctx.calls)
        snapshot_at = h.events.index(("snapshot", "pre-jev-curator-apply"))
        first_dispatch = next(i for i, event in enumerate(h.events) if event[0] == "dispatch")
        self.assertLess(snapshot_at, first_dispatch)

    def test_failed_or_unavailable_snapshot_refuses_with_zero_dispatches(self):
        cases = {
            "raises": {"snapshot_error": RuntimeError("disk full")},
            "unavailable": {"snapshot": None},
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label):
                with EngineHarness(mode="apply", **kwargs) as h:
                    result = h.engine.apply([make_plan()])
                self.assertFalse(result["ok"])
                self.assertIn("snapshot", result["error"])
                self.assertEqual(h.ctx.calls, [])


# --- 8. every real write routes through ctx.dispatch_tool("skill_manage", ...) ----------

class WritePathTests(unittest.TestCase):
    def test_apply_routes_deletes_through_dispatch_tool_under_background_review(self):
        inventory = default_inventory()
        plan = make_plan(canonical="alpha", absorbed=("beta", "gamma"))
        with EngineHarness(mode="apply", inventory=inventory) as h:
            result = h.engine.apply([plan])
            origin_after = h.provenance.get_current_write_origin()
            attended_after = h.provenance.get_review_attended()

        self.assertTrue(result["ok"])
        self.assertEqual([row["skill"] for row in result["applied"]], ["beta", "gamma"])
        self.assertTrue(all(row["result"]["success"] for row in result["applied"]))

        manages = [call for call in h.ctx.calls if call["tool"] == "skill_manage"]
        self.assertEqual(
            [(call["args"]["action"], call["args"]["name"], call["args"]["absorbed_into"])
             for call in manages],
            [("delete", "beta", "alpha"), ("delete", "gamma", "alpha")])
        for call in manages:
            self.assertEqual(set(call["args"]), {"action", "name", "absorbed_into"},
                             "writes must use exactly the skill_manage delete payload")
        self.assertEqual(h.ctx.calls, manages,
                         "apply must not mutate telemetry through a skill_view preflight")

        for call in h.ctx.calls:
            self.assertEqual(call["origin"], "background_review")
            self.assertTrue(call["attended"])
        self.assertEqual(origin_after, "foreground", "write origin must be restored")
        self.assertFalse(attended_after, "review-attended flag must be restored")

    def test_failed_delete_stops_the_run_and_restores_provenance(self):
        plan = make_plan(canonical="alpha", absorbed=("beta", "gamma"))
        manage_results = (json.dumps({"success": False, "error": "guarded"}),)
        with EngineHarness(mode="apply", manage_results=manage_results) as h:
            result = h.engine.apply([plan])
            origin_after = h.provenance.get_current_write_origin()
            attended_after = h.provenance.get_review_attended()

        self.assertFalse(result["ok"])
        self.assertIn("skill_manage failed for 'beta'", result["error"])
        self.assertEqual([row["skill"] for row in result["applied"]], ["beta"])
        manages = [call for call in h.ctx.calls if call["tool"] == "skill_manage"]
        self.assertEqual(len(manages), 1, "the second delete must not run after a failure")
        self.assertEqual(origin_after, "foreground")
        self.assertFalse(attended_after)

    def test_staged_write_stops_without_reporting_success(self):
        plan = make_plan(canonical="alpha", absorbed=("beta", "gamma"))
        manage_results = (json.dumps({"success": True, "staged": True, "pending_id": "pending-1"}),)
        with EngineHarness(mode="apply", manage_results=manage_results) as h:
            result = h.engine.apply([plan])

        self.assertFalse(result["ok"])
        self.assertIn("staged", result["error"])
        self.assertEqual(result["applied"], [])
        self.assertEqual([row["skill"] for row in result["staged"]], ["beta"])
        self.assertTrue(result["staged"][0]["pending_discarded"])
        self.assertEqual(h.discarded_pending, [("skills", "pending-1")])
        self.assertEqual(len([call for call in h.ctx.calls if call["tool"] == "skill_manage"]), 1)

    def test_staged_write_discard_failure_is_reported_not_raised(self):
        manage_results = (json.dumps({"success": True, "staged": True,
                                      "pending_id": "pending-1"}),)
        with EngineHarness(mode="apply", manage_results=manage_results) as h:
            approval = sys.modules["tools.write_approval"]
            approval.discard_pending = mock.Mock(side_effect=OSError("disk"))
            result = h.engine.apply([make_plan()])
        self.assertFalse(result["ok"])
        self.assertFalse(result["staged"][0]["pending_discarded"])
        self.assertIn("discard it manually", result["error"])

    def test_digest_is_rechecked_immediately_before_each_delete(self):
        good = default_inventory()
        changed = [replace(item, digest="changed") if item.name == "beta" else item
                   for item in good]
        with EngineHarness(mode="apply", inventory=good) as h:
            with mock.patch.object(engine_module, "collect_inventory",
                                   side_effect=[good, changed]):
                result = h.engine.apply([make_plan()])
        self.assertFalse(result["ok"])
        self.assertIn("immediately before", result["error"])
        self.assertEqual(h.ctx.calls, [])

    def test_dispatch_exception_returns_recovery_report(self):
        with EngineHarness(mode="apply") as h:
            with mock.patch.object(h.ctx, "dispatch_tool", side_effect=RuntimeError("boom")):
                result = h.engine.apply([make_plan()])
        self.assertFalse(result["ok"])
        self.assertIn("skill_manage raised", result["error"])
        self.assertEqual(result["recovery"]["snapshot"], "snap-1")

    def test_success_without_archive_marker_is_refused(self):
        manage_results = (json.dumps({"success": True, "message": "deleted"}),)
        with EngineHarness(mode="apply", manage_results=manage_results) as h:
            result = h.engine.apply([make_plan()])
        self.assertFalse(result["ok"])
        self.assertIn("recoverable archive", result["error"])

    def test_archive_marker_is_verified_against_inventory(self):
        with EngineHarness(mode="apply", auto_archive=False) as h:
            result = h.engine.apply([make_plan()])
        self.assertFalse(result["ok"])
        self.assertIn("still present", result["error"])

    def test_canonical_digest_is_rechecked_after_each_archive(self):
        with EngineHarness(mode="apply", post_archive_digests={"alpha": "changed"}) as h:
            result = h.engine.apply([make_plan()])
        self.assertFalse(result["ok"])
        self.assertIn("canonical changed", result["error"])
        self.assertEqual(result["recovery"]["restore_commands"],
                         ["hermes curator restore beta"])

    def test_partial_apply_returns_restore_commands(self):
        plan = make_plan(canonical="alpha", absorbed=("beta", "gamma"))
        manage_results = (
            json.dumps({"success": True, "_archived": True}),
            json.dumps({"success": False, "error": "guarded"}),
        )
        with EngineHarness(mode="apply", manage_results=manage_results) as h:
            result = h.engine.apply([plan])
        self.assertFalse(result["ok"])
        self.assertEqual(result["recovery"]["snapshot"], "snap-1")
        self.assertEqual(result["recovery"]["restore_commands"],
                         ["hermes curator restore beta"])


# --- adjacent apply gates ---------------------------------------------------------------

class ApplyGateTests(unittest.TestCase):
    def test_overlapping_or_chained_plans_are_refused_before_backup(self):
        cases = {
            "chain": [make_plan(canonical="alpha", absorbed=("beta",), plan_id="merge-1"),
                      make_plan(canonical="beta", absorbed=("gamma",), plan_id="merge-2")],
            "shared source": [make_plan(canonical="alpha", absorbed=("beta",), plan_id="merge-1"),
                              make_plan(canonical="gamma", absorbed=("beta",), plan_id="merge-2")],
        }
        for label, plans in cases.items():
            with self.subTest(label=label):
                with EngineHarness(mode="apply") as h:
                    result = h.engine.apply(plans)
                self.assertFalse(result["ok"])
                self.assertIn("overlap", result["error"])
                self.assertEqual(h.ctx.calls, [])
                self.assertEqual(h.snapshot_calls, [])

    def test_enabled_write_approval_refuses_before_backup(self):
        with EngineHarness(mode="apply", approval_enabled=True) as h:
            result = h.engine.apply([make_plan()])

        self.assertFalse(result["ok"])
        self.assertIn("write_approval", result["error"])
        self.assertEqual(h.ctx.calls, [])
        self.assertEqual(h.snapshot_calls, [])

    def test_unverifiable_write_approval_gate_refuses(self):
        with EngineHarness(mode="apply", approval_module=False) as h:
            result = h.engine.apply([make_plan()])

        self.assertFalse(result["ok"])
        self.assertIn("could not verify", result["error"])
        self.assertEqual(h.ctx.calls, [])


if __name__ == "__main__":
    unittest.main()
