"""Guard tests: throwaway HERMES_HOME, faked inventory, no real profile writes, no network."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plugin import guard, state
from plugin.graph import build_plans, pair_key
from plugin.models import MergePlan, RelationJudgment, Settings, SkillArtifact


def make_plan(*, plan_id="merge-0001", canonical="umbrella", canonical_digest="d-umb",
              absorbed=None, status="validated", blockers=(), edges=None):
    absorbed_map = {"old": "d-old"} if absorbed is None else dict(absorbed)
    if edges is None:
        edges = tuple(sorted(pair_key(name, canonical) for name in absorbed_map))
    return MergePlan(plan_id=plan_id, canonical=canonical, canonical_digest=canonical_digest,
                     absorbed=tuple(sorted(absorbed_map)), absorbed_digests=absorbed_map,
                     relation_keys=tuple(edges), status=status, blockers=tuple(blockers))


class GuardTestCase(unittest.TestCase):
    ENV_KEYS = ("HERMES_HOME", "JEV_CURATOR_AUDIT_MAX_BYTES")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.root = self.home / "jev-curator"
        saved = {key: os.environ.get(key) for key in self.ENV_KEYS}
        self.addCleanup(self._restore_env, saved)
        os.environ["HERMES_HOME"] = str(self.home)
        os.environ.pop("JEV_CURATOR_AUDIT_MAX_BYTES", None)

    def _restore_env(self, saved):
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # -- seams -------------------------------------------------------------------

    def package(self, name, digest, protected=(), files=()):
        path = self.root / "skills" / name
        path.mkdir(parents=True, exist_ok=True)
        for relative in files:
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        return SkillArtifact(name=name, path=path, description="", text="", digest=digest,
                             protected_reasons=tuple(protected))

    def inventory(self, *artifacts, missing=False):
        table = {item.name: item for item in artifacts}

        def lookup(names):
            return None if missing else {name: table[name] for name in names if name in table}

        return mock.patch.object(guard, "_current_artifacts", side_effect=lookup)

    def call(self, args, *, mode="guard", background=True, tool="skill_manage", settings=None):
        source = settings if settings is not None else Settings(mode=mode)
        with mock.patch.object(guard, "_background_review", return_value=background):
            return guard.pre_tool_call_guard(source, tool, args)

    # -- assertions --------------------------------------------------------------

    def assertBlock(self, result, needle=""):
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("action"), "block")
        message = result.get("message")
        self.assertIsInstance(message, str)
        self.assertTrue(message.strip())
        self.assertLessEqual(len(message), guard._MAX_MESSAGE_CHARS)
        self.assertNotIn("\n", message)
        if needle:
            self.assertIn(needle, message)

    def assertPass(self, result):
        self.assertIsNone(result)


class ModeTests(GuardTestCase):
    def test_off_and_observe_never_interfere(self):
        for mode in ("off", "observe"):
            with self.subTest(mode=mode):
                with self.inventory(self.package("old", "d-old")):
                    self.assertPass(self.call({"action": "delete", "name": "old",
                                               "absorbed_into": "umbrella"}, mode=mode))
                    self.assertPass(self.call({"action": "patch", "name": "old"}, mode=mode))

    def test_mapping_and_unknown_modes_are_inert(self):
        with self.inventory(self.package("old", "d-old")):
            self.assertPass(self.call({"action": "delete", "name": "old"}, settings={"mode": "observe"}))
            self.assertPass(self.call({"action": "delete", "name": "old"}, settings={"mode": "bogus"}))
            with mock.patch.object(guard, "_background_review", return_value=True):
                self.assertPass(guard.pre_tool_call_guard(None, "skill_manage",
                                                          {"action": "delete", "name": "old"}))

    def test_guard_and_apply_fail_closed_without_plans(self):
        for mode in ("guard", "apply"):
            with self.subTest(mode=mode):
                with self.inventory(self.package("old", "d-old")):
                    self.assertBlock(self.call({"action": "delete", "name": "old",
                                                "absorbed_into": "umbrella"}, mode=mode),
                                     "no authorized merge plan")


class ProvenanceTests(GuardTestCase):
    def test_foreground_calls_are_never_blocked(self):
        with self.inventory(self.package("old", "d-old")):
            self.assertPass(self.call({"action": "delete", "name": "old", "absorbed_into": "u"},
                                      background=False))
            self.assertPass(self.call({"operations": [{"action": "patch", "name": "old"}]},
                                      background=False))
            self.assertPass(self.call({"action": "write_file", "name": "old",
                                       "file_path": "references/x.md"}, background=False))

    def test_background_calls_fail_closed(self):
        with self.inventory(self.package("old", "d-old")):
            self.assertBlock(self.call({"action": "delete", "name": "old", "absorbed_into": "u"}))

    def test_real_provenance_contextvar_drives_the_split(self):
        try:
            from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
        except ImportError:  # standalone checkout without the Hermes core on sys.path
            self.skipTest("tools.skill_provenance unavailable")
        args = {"action": "delete", "name": "old", "absorbed_into": "umbrella"}
        with self.inventory(self.package("old", "d-old")):
            token = set_current_write_origin("background_review")
            try:
                self.assertBlock(guard.pre_tool_call_guard(Settings(mode="guard"), "skill_manage", args))
            finally:
                reset_current_write_origin(token)
            self.assertPass(guard.pre_tool_call_guard(Settings(mode="guard"), "skill_manage", args))


class ShapeTests(GuardTestCase):
    def test_other_tools_and_harmless_ops_pass(self):
        with self.inventory(self.package("old", "d-old")):
            for tool in ("skill_view", "read_file", "terminal", "skills_list", ""):
                self.assertPass(self.call({"action": "delete", "name": "old"}, tool=tool))
            self.assertPass(self.call({"action": "create", "name": "new", "content": "x"}))
            self.assertPass(self.call({"action": "write_file", "name": "new", "file_path": "references/a.md"}))
            self.assertPass(self.call({"operations": [{"action": "create", "name": "new"}]}))
            self.assertPass(self.call({"action": "mystery", "name": "old"}))

    def test_hostile_args_never_raise(self):
        for args in (None, "delete everything", 17, [], {"operations": "nope"},
                     {"operations": [None, 3, "x"]}, {"operations": []},
                     {"action": "delete"}, {"action": "delete", "name": 5},
                     {"action": ["delete"], "name": {"a": 1}}):
            with self.subTest(args=args):
                result = self.call(args)
                self.assertTrue(result is None or (isinstance(result, dict)
                                                   and result.get("action") == "block"))

    def test_flat_and_batch_shapes_normalize_identically(self):
        with self.inventory(self.package("old", "d-old")):
            flat = self.call({"action": "patch", "name": "old", "old_string": "a", "new_string": "b"})
            batch = self.call({"operations": [{"action": "patch", "name": "old",
                                               "old_string": "a", "new_string": "b"}]})
        self.assertBlock(flat, "content mutation is not authorized")
        self.assertEqual(flat, batch)

    def test_batch_blocks_when_any_op_is_destructive(self):
        with self.inventory(self.package("old", "d-old")):
            self.assertBlock(self.call({"operations": [
                {"action": "create", "name": "fresh"},
                {"action": "remove_file", "name": "old", "file_path": "references/a.md"},
            ]}), "content mutation is not authorized")

    def test_batch_default_name_is_used(self):
        with self.inventory(self.package("old", "d-old")):
            self.assertBlock(self.call({"name": "old", "operations": [{"action": "patch"}]}),
                             "'old'")

    def test_new_file_write_fails_open_and_overwrite_does_not(self):
        target = self.package("old", "d-old", files=("references/api.md",))
        with self.inventory(target):
            self.assertPass(self.call({"action": "write_file", "name": "old",
                                       "file_path": "references/new.md"}))
            self.assertBlock(self.call({"action": "write_file", "name": "old",
                                        "file_path": "references/api.md"}),
                             "content mutation is not authorized")
            self.assertBlock(self.call({"action": "write_file", "name": "old",
                                        "file_path": "../escape.md"}), "content mutation is not authorized")

    def test_remove_file_and_edit_are_destructive(self):
        with self.inventory(self.package("old", "d-old")):
            self.assertBlock(self.call({"action": "remove_file", "name": "old",
                                        "file_path": "references/a.md"}))
            self.assertBlock(self.call({"action": "edit", "name": "old", "content": "x"}))

    def test_inventory_reads_only_requested_packages(self):
        artifacts = [self.package("old", "d-old"), self.package("umbrella", "d-umb")]
        wanted = {"old", "umbrella"}
        with mock.patch.object(guard, "collect_inventory", return_value=artifacts) as collect:
            result = guard._current_artifacts(wanted)
        self.assertEqual(set(result or {}), wanted)
        collect.assert_called_once_with(include_unmanaged=True, names=wanted)

    def test_path_alias_overwrite_is_blocked_when_frontmatter_name_differs(self):
        package = self.home / "skills" / "directory-name"
        package.mkdir(parents=True)
        (package / "SKILL.md").write_text("original", encoding="utf-8")
        artifact = SkillArtifact(
            name="frontmatter-name", path=package, description="", text="", digest="d",
            protected_reasons=("name-path-mismatch",),
        )
        with mock.patch.object(guard, "collect_inventory", return_value=[artifact]):
            found = guard._current_artifacts({"directory-name"})
        self.assertEqual((found or {}).get("directory-name"), artifact)
        with mock.patch.object(guard, "_current_artifacts", return_value=found):
            self.assertBlock(self.call({
                "action": "write_file", "name": "directory-name", "file_path": "SKILL.md",
            }), "protected")

    def test_categorized_path_alias_is_resolved(self):
        package = self.home / "skills" / "mlops" / "axolotl"
        package.mkdir(parents=True)
        artifact = SkillArtifact(
            name="axolotl", path=package, description="", text="", digest="d",
        )
        with mock.patch.object(guard, "collect_inventory", return_value=[artifact]):
            found = guard._current_artifacts({"mlops/axolotl"})
        self.assertEqual((found or {}).get("mlops/axolotl"), artifact)


class OutageMatrixTests(GuardTestCase):
    def args(self):
        return {"action": "delete", "name": "old", "absorbed_into": "umbrella"}

    def test_missing_store_fails_closed(self):
        with self.inventory(self.package("old", "d-old")):
            self.assertBlock(self.call(self.args()), "no authorized merge plan is installed")

    def test_corrupt_store_fails_closed(self):
        guard.plans_path().parent.mkdir(parents=True, exist_ok=True)
        guard.plans_path().write_text("{not json", encoding="utf-8")
        with self.inventory(self.package("old", "d-old")):
            self.assertBlock(self.call(self.args()), "unreadable or malformed")

    def test_wrong_shapes_fail_closed(self):
        for payload in (["nope"], {"plans": "nope"}, {"plans": {}}, {"plans": {"a": {"plan_id": ""}}}):
            with self.subTest(payload=payload):
                state.write_json(guard.plans_path(), payload)
                with self.inventory(self.package("old", "d-old")):
                    self.assertBlock(self.call(self.args()))

    def test_malformed_entries_are_dropped_not_trusted(self):
        entry = {"plan_id": "merge-x", "status": "validated", "canonical": "umbrella",
                 "canonical_digest": "d-umb", "absorbed": {"old": "d-old"}}
        state.write_json(guard.plans_path(), {"version": 1, "plans": {"x": entry}})
        with self.inventory(self.package("old", "d-old")):
            self.assertBlock(self.call(self.args()), "unreadable or malformed")

    def test_inventory_outage_fails_closed(self):
        with mock.patch.object(guard, "_current_artifacts", return_value=None):
            self.assertBlock(self.call(self.args()), "inventory is unreadable")
        with self.inventory(self.package("old", "d-old"), missing=True):
            self.assertBlock(self.call(self.args()), "inventory is unreadable")

    def test_stale_hashes_fail_closed(self):
        self.assertIsNotNone(guard.install_authorized_plan(make_plan()))
        with self.inventory(self.package("old", "d-new"), self.package("umbrella", "d-umb")):
            self.assertBlock(self.call(self.args()), "stale hash")

    def test_matching_plan_authorizes(self):
        self.assertIsNotNone(guard.install_authorized_plan(make_plan()))
        with self.inventory(self.package("old", "d-old"), self.package("umbrella", "d-umb")):
            self.assertPass(self.call(self.args()))
            self.assertBlock(self.call({"action": "patch", "name": "umbrella"}),
                             "content mutation is not authorized")
            self.assertBlock(self.call({"action": "patch", "name": "old"}),
                             "content mutation is not authorized")

    def test_plan_for_another_skill_does_not_cover(self):
        self.assertIsNotNone(guard.install_authorized_plan(
            make_plan(plan_id="merge-other", canonical="other", canonical_digest="d-o",
                      absorbed={"x": "d-x"})))
        with self.inventory(self.package("old", "d-old"), self.package("umbrella", "d-umb")):
            self.assertBlock(self.call(self.args()), "no authorized merge plan absorbs")


class DeletePlanTests(GuardTestCase):
    def setUp(self):
        super().setUp()
        self.assertIsNotNone(guard.install_authorized_plan(make_plan()))
        self.inv = self.inventory(self.package("old", "d-old"), self.package("umbrella", "d-umb"))

    def test_exact_pair_authorizes(self):
        with self.inv:
            self.assertPass(self.call({"action": "delete", "name": "old", "absorbed_into": "umbrella"}))

    def test_mismatched_absorbed_into_blocks(self):
        with self.inv:
            self.assertBlock(self.call({"action": "delete", "name": "old", "absorbed_into": "other"}),
                             "does not match the authorized plan (expected 'umbrella')")

    def test_missing_or_self_absorbed_into_blocks(self):
        with self.inv:
            self.assertBlock(self.call({"action": "delete", "name": "old"}), "absorbed_into")
            self.assertBlock(self.call({"action": "delete", "name": "old", "absorbed_into": "  "}),
                             "absorbed_into")
            self.assertBlock(self.call({"action": "delete", "name": "old", "absorbed_into": "old"}),
                             "cannot equal")

    def test_missing_umbrella_blocks(self):
        with self.inventory(self.package("old", "d-old")):
            self.assertBlock(self.call({"action": "delete", "name": "old",
                                        "absorbed_into": "umbrella"}), "not in the managed inventory")

    def test_stale_umbrella_blocks(self):
        with self.inventory(self.package("old", "d-old"), self.package("umbrella", "d-changed")):
            self.assertBlock(self.call({"action": "delete", "name": "old",
                                        "absorbed_into": "umbrella"}), "stale hash")

    def test_canonical_is_not_deletable_through_its_own_plan(self):
        with self.inv:
            self.assertBlock(self.call({"action": "delete", "name": "umbrella",
                                        "absorbed_into": "old"}), "no authorized merge plan absorbs")

    def test_direct_edge_is_required(self):
        # A hand-written store entry whose member lacks its own edge to the canonical is
        # dropped as malformed rather than trusted as a direct-edge authorization.
        entry = {"plan_id": "merge-x", "status": "validated", "canonical": "umbrella",
                 "canonical_digest": "d-umb", "absorbed": {"old": "d-old"}, "edges": []}
        state.write_json(guard.plans_path(), {"version": 1, "plans": {"x": entry}})
        with self.inv:
            self.assertBlock(self.call({"action": "delete", "name": "old",
                                        "absorbed_into": "umbrella"}))
        self.assertIsNone(guard.install_authorized_plan(make_plan(edges=())))


class ProtectedTests(GuardTestCase):
    def test_protected_targets_block_even_with_a_valid_plan(self):
        self.assertIsNotNone(guard.install_authorized_plan(make_plan()))
        with self.inventory(self.package("old", "d-old", protected=("pinned",)),
                            self.package("umbrella", "d-umb")):
            self.assertBlock(self.call({"action": "delete", "name": "old",
                                        "absorbed_into": "umbrella"}), "protected (pinned)")
        with self.inventory(self.package("old", "d-old"),
                            self.package("umbrella", "d-umb", protected=("bundled", "pinned"))):
            self.assertBlock(self.call({"action": "delete", "name": "old",
                                        "absorbed_into": "umbrella"}), "protected (bundled,pinned)")

    def test_protected_write_target_blocks(self):
        self.assertIsNotNone(guard.install_authorized_plan(make_plan()))
        with self.inventory(self.package("umbrella", "d-umb", protected=("cron-referenced",)),
                            self.package("old", "d-old")):
            self.assertBlock(self.call({"action": "patch", "name": "umbrella"}), "protected")


class InstallTests(GuardTestCase):
    def test_install_roundtrip(self):
        plan = make_plan()
        self.assertEqual(guard.install_authorized_plan(plan), "merge-0001")
        entries = guard.load_authorized_plans()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["canonical"], "umbrella")
        self.assertEqual(entries[0]["absorbed"], {"old": "d-old"})
        self.assertEqual(entries[0]["edges"], ["old::umbrella"])
        self.assertEqual(stat.S_IMODE(guard.plans_path().stat().st_mode), 0o600)
        self.assertTrue(str(guard.plans_path()).startswith(str(self.root)))

    def test_install_accepts_asdict_mapping(self):
        self.assertEqual(guard.install_authorized_plan(dataclasses.asdict(make_plan())), "merge-0001")
        self.assertEqual(len(guard.load_authorized_plans()), 1)

    def test_replace_removes_old_authority_and_keeps_only_valid_current_plans(self):
        self.assertEqual(guard.install_authorized_plan(make_plan()), "merge-0001")
        current = make_plan(plan_id="merge-0002", canonical="target",
                            canonical_digest="d-target", absorbed={"source": "d-source"},
                            edges=("source::target",))
        self.assertEqual(guard.replace_authorized_plans([current, make_plan(status="blocked")]),
                         ["merge-0002"])
        entries = guard.load_authorized_plans()
        self.assertEqual([row["plan_id"] for row in entries], ["merge-0002"])
        self.assertEqual(guard.replace_authorized_plans([]), [])
        self.assertEqual(guard.load_authorized_plans(), [])

    def test_install_refuses_invalid_plans(self):
        bad = (
            None, "merge-x", 42, {},
            make_plan(status="blocked"),
            make_plan(blockers=("protected:old:pinned",)),
            make_plan(status="noop"),
            make_plan(plan_id=""),
            make_plan(canonical=""),
            make_plan(canonical_digest=""),
            make_plan(absorbed={}),
            make_plan(absorbed={"": "d"}),
            make_plan(absorbed={"old": ""}),
            make_plan(absorbed={"umbrella": "d-umb"}),
            make_plan(edges=()),
            make_plan(edges=("old::other",)),
        )
        for plan in bad:
            with self.subTest(plan=plan):
                self.assertIsNone(guard.install_authorized_plan(plan))
        self.assertEqual(guard.load_authorized_plans(), [])
        self.assertFalse(guard.plans_path().exists())

    def test_install_upserts_and_caps(self):
        self.assertEqual(guard.install_authorized_plan(make_plan()), "merge-0001")
        self.assertEqual(guard.install_authorized_plan(make_plan(canonical_digest="d-2")), "merge-0001")
        entries = guard.load_authorized_plans()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["canonical_digest"], "d-2")
        for index in range(guard._MAX_PLANS + 3):
            guard.install_authorized_plan(make_plan(plan_id=f"merge-{index:04d}"))
        entries = guard.load_authorized_plans()
        self.assertEqual(len(entries), guard._MAX_PLANS)
        plan_ids = {entry["plan_id"] for entry in entries}
        self.assertIn(f"merge-{guard._MAX_PLANS + 2:04d}", plan_ids)  # newest kept
        self.assertNotIn("merge-0001", plan_ids)  # oldest trimmed

    def test_install_recovers_from_a_corrupt_store(self):
        guard.plans_path().parent.mkdir(parents=True, exist_ok=True)
        guard.plans_path().write_text("{broken", encoding="utf-8")
        self.assertEqual(guard.install_authorized_plan(make_plan()), "merge-0001")
        self.assertEqual(len(guard.load_authorized_plans()), 1)

    def test_install_never_raises_on_hostile_input(self):
        class Hostile:
            @property
            def absorbed_digests(self):
                raise RuntimeError("boom")

        self.assertIsNone(guard.install_authorized_plan(Hostile()))
        self.assertIsNone(guard.install_authorized_plan({"plan_id": "x", "absorbed_digests": object()}))


class MessageTests(GuardTestCase):
    def test_blocks_are_bounded_single_line_and_audited(self):
        hostile = self.package("bad\nname " + "z" * 80, "d-old")
        with self.inventory(hostile):
            result = self.call({"action": "patch", "name": hostile.name})
        self.assertBlock(result)
        self.assertNotIn("bad\nname", (result or {}).get("message", ""))
        rows = [json.loads(line) for line in state.audit_log_path().read_text(encoding="utf-8").splitlines()]
        self.assertIn("guard_block", [row["event"] for row in rows])

    def test_install_is_audited(self):
        guard.install_authorized_plan(make_plan())
        rows = [json.loads(line) for line in state.audit_log_path().read_text(encoding="utf-8").splitlines()]
        self.assertIn("guard_plan_install", [row["event"] for row in rows])

    def test_only_delete_refusals_suggest_refreshing_authority(self):
        with self.inventory(self.package("old", "d-old")):
            patch_result = self.call({"action": "patch", "name": "old"})
            delete_result = self.call({"action": "delete", "name": "old",
                                       "absorbed_into": "umbrella"})
        self.assertNotIn("Install or refresh", patch_result["message"])
        self.assertIn("Install or refresh", delete_result["message"])


class AdapterTests(GuardTestCase):
    def test_hook_adapter_accepts_extra_kwargs(self):
        hook = guard.make_pre_tool_call_hook(lambda: Settings(mode="guard"))
        with self.inventory(self.package("old", "d-old")):
            with mock.patch.object(guard, "_background_review", return_value=True):
                result = hook(tool_name="skill_manage", args={"action": "patch", "name": "old"},
                              session_id="s", turn_id="t", task_id="k", tool_call_id="c",
                              middleware_trace=[])
                self.assertBlock(result)
                self.assertPass(hook(tool_name="skill_view", args={"name": "old"}, session_id="s"))
                self.assertPass(hook())

    def test_service_like_source_and_raising_factory(self):
        class Service:
            settings = Settings(mode="apply")

        with self.inventory(self.package("old", "d-old")):
            with mock.patch.object(guard, "_background_review", return_value=True):
                self.assertBlock(guard.pre_tool_call_guard(Service(), "skill_manage",
                                                           {"action": "patch", "name": "old"}))
        with mock.patch.object(guard, "_background_review", return_value=True):
            def boom():
                raise RuntimeError("config unreadable")

            self.assertPass(guard.pre_tool_call_guard(boom, "skill_manage",
                                                      {"action": "patch", "name": "old"}))


class NoNetworkTests(GuardTestCase):
    def test_decision_makes_no_network_calls(self):
        guard.install_authorized_plan(make_plan())
        with self.inventory(self.package("old", "d-old"), self.package("umbrella", "d-umb")):
            with mock.patch("socket.socket", side_effect=AssertionError("network use")):
                self.assertPass(self.call({"action": "delete", "name": "old",
                                           "absorbed_into": "umbrella"}))
                self.assertBlock(self.call({"action": "delete", "name": "old",
                                            "absorbed_into": "other"}))


class RealPlanIntegrationTests(GuardTestCase):
    def test_graph_built_plan_installs_and_authorizes_its_delete(self):
        umbrella = self.package("umbrella", "d-umb")
        old = self.package("old", "d-old")
        judgment = RelationJudgment(
            a="old", b="umbrella", a_digest="d-old", b_digest="d-umb",
            relation="a_subset_of_b", confidence=0.95, probabilities={"a_subset_of_b": 0.95},
            coverage=0.9, preservation_a_in_b=0.95, preservation_b_in_a=0.5, conflict=0.0,
            contract_version="skill-relations-v1")
        applicable = [plan for plan in build_plans([umbrella, old], [judgment]) if plan.applicable]
        self.assertEqual(len(applicable), 1)
        plan = applicable[0]
        self.assertEqual((plan.canonical, plan.absorbed), ("umbrella", ("old",)))
        self.assertEqual(guard.install_authorized_plan(plan), plan.plan_id)
        with self.inventory(umbrella, old):
            self.assertPass(self.call({"action": "delete", "name": "old", "absorbed_into": "umbrella"}))
        with self.inventory(umbrella, self.package("old", "d-changed")):
            self.assertBlock(self.call({"action": "delete", "name": "old",
                                        "absorbed_into": "umbrella"}), "stale hash")


if __name__ == "__main__":
    unittest.main()
