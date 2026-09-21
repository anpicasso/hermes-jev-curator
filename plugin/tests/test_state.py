"""Storage tests: every write lands in a throwaway HERMES_HOME, never the real home."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from plugin import state
from plugin.models import MergePlan, RelationJudgment


def make_judgment(a: str = "alpha", b: str = "beta", **overrides) -> RelationJudgment:
    base = RelationJudgment(
        a=a, b=b, a_digest=f"{a}-d1", b_digest=f"{b}-d2",
        relation="same_class", confidence=0.8,
        probabilities={"same_class": 0.8, "unrelated": 0.2},
        coverage=0.9, preservation_a_in_b=0.5, preservation_b_in_a=0.5, conflict=0.1,
        contract_version="skill-relations-v1", raw_model="jev-latest",
    )
    return replace(base, **overrides) if overrides else base


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class StateTestCase(unittest.TestCase):
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


class PathTests(StateTestCase):
    def test_root_follows_active_hermes_home(self):
        self.assertEqual(state.state_root(), self.root)
        fake = self.home / "hermes-profile"
        os.environ["HERMES_HOME"] = str(fake)
        self.assertEqual(state.state_root(), fake / "jev-curator")
        self.assertNotIn("skills", state.state_root().parts)
        for path in (state.audit_log_path(), state.state_path(), state.relations_path(),
                     state.lock_path(), state.reports_dir()):
            self.assertTrue(str(path).startswith(str(fake / "jev-curator") + os.sep))
            self.assertNotIn(path.name, {".usage.json", ".curator_state"})


class AuditTests(StateTestCase):
    def rows(self):
        text = state.audit_log_path().read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def test_appends_secure_jsonl(self):
        self.assertTrue(state.audit("run.start", mode="observe", pair={"a": "x", "b": ["y", 3]}))
        self.assertTrue(state.audit("run.end", ok=True))
        rows = self.rows()
        self.assertEqual([row["event"] for row in rows], ["run.start", "run.end"])
        self.assertEqual(rows[0]["mode"], "observe")
        self.assertEqual(rows[0]["pair"], {"a": "x", "b": ["y", 3]})
        self.assertIsInstance(rows[0]["ts"], float)
        self.assertEqual(file_mode(state.audit_log_path()), 0o600)
        self.assertEqual(file_mode(state.state_root()) & 0o077, 0)

    def test_redacts_secrets(self):
        secret = "sk-abc123XYZdef456ghi789"
        bearer = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
        self.assertTrue(state.audit("call", headers={"Authorization": f"Bearer {bearer}"},
                                    api_key=secret, note="password: swordfish99"))
        raw = state.audit_log_path().read_text(encoding="utf-8")
        for leak in (secret, bearer, "swordfish99"):
            self.assertNotIn(leak, raw)

    def test_redacts_shell_and_cookie_credentials(self):
        secrets = ("cookievalue123", "passphrase123", "hunter2")
        payload = (
            "Cookie: session=cookievalue123\n"
            "curl -u user:passphrase123 https://example.invalid\n"
            "--password hunter2"
        )
        self.assertTrue(state.audit("credential-shapes", payload=payload))
        state.write_report("credential-shapes", {"payload": payload})
        persisted = (
            state.audit_log_path().read_text(encoding="utf-8")
            + (state.reports_dir() / "credential-shapes.json").read_text(encoding="utf-8")
        )
        for secret in secrets:
            self.assertNotIn(secret, persisted)

    def test_fallback_redactor_masks_when_canonical_is_absent(self):
        text = state._FALLBACK_SECRET_RE.sub(
            "[redacted]", "MY_API_KEY=abcdef123456 password: swordfish99 sk-abc123XYZdef456ghi789")
        for leak in ("abcdef123456", "swordfish99", "sk-abc123XYZdef456ghi789"):
            self.assertNotIn(leak, text)
        self.assertEqual(state._FALLBACK_SECRET_RE.sub("[redacted]", "nothing sensitive here"),
                         "nothing sensitive here")

    def test_never_raises_on_hostile_payloads_and_bad_paths(self):
        loop: list = []
        loop.append(loop)
        self.assertTrue(state.audit("weird", obj=object(), loop=loop, blob=b"\x00\xff"))
        self.assertEqual(self.rows()[-1]["event"], "weird")
        target = self.home / "outside.jsonl"
        target.write_text("", encoding="utf-8")
        state.audit_log_path().unlink()
        state.audit_log_path().symlink_to(target)
        self.assertFalse(state.audit("after-symlink"))  # O_NOFOLLOW refuses the symlink
        self.assertEqual(target.read_text(encoding="utf-8"), "")
        os.environ["HERMES_HOME"] = str(self.home / "blocked")
        (self.home / "blocked").write_text("not a dir", encoding="utf-8")
        self.assertFalse(state.audit("blocked"))

    def test_rotation_keeps_one_generation(self):
        os.environ["JEV_CURATOR_AUDIT_MAX_BYTES"] = "500"
        for index in range(20):
            self.assertTrue(state.audit("row", index=index, payload="x" * 200))
        log = state.audit_log_path()
        rotated = log.with_suffix(log.suffix + ".1")
        self.assertTrue(rotated.is_file())
        self.assertEqual(file_mode(rotated), 0o600)
        self.assertLessEqual(log.stat().st_size, 500)
        self.assertEqual(sorted(p.name for p in self.root.glob("audit.jsonl*")),
                         ["audit.jsonl", "audit.jsonl.1"])
        rotated_rows = [json.loads(line) for line in rotated.read_text().splitlines() if line.strip()]
        live_rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        self.assertEqual(live_rows[-1]["index"], 19)
        self.assertLess(rotated_rows[0]["index"], live_rows[0]["index"])


class JsonTests(StateTestCase):
    def test_atomic_roundtrip_and_corruption_tolerance(self):
        target = self.root / "nested" / "data.json"
        state.write_json(target, {"b": 1, "a": [1, 2]})
        self.assertEqual(state.read_json(target), {"b": 1, "a": [1, 2]})
        self.assertEqual(file_mode(target), 0o600)
        self.assertEqual(list(target.parent.glob("*.tmp")), [])
        target.write_text("{not json", encoding="utf-8")
        self.assertEqual(state.read_json(target, {"fallback": True}), {"fallback": True})
        self.assertEqual(state.read_json(self.root / "missing.json", "d"), "d")

    def test_read_json_refuses_symlinks_and_oversized_files(self):
        outside = self.home / "outside.json"
        outside.write_text('{"secret":"must-not-be-read"}', encoding="utf-8")
        link = self.root / "link.json"
        self.root.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside)
        self.assertEqual(state.read_json(link, {"safe": True}), {"safe": True})

        large = self.root / "large.json"
        large.write_bytes(b" " * (state._MAX_JSON_BYTES + 1))
        self.assertEqual(state.read_json(large, "bounded"), "bounded")

    @unittest.skipUnless(hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK"),
                         "requires POSIX nonblocking FIFOs")
    def test_read_json_refuses_fifo_without_blocking(self):
        self.root.mkdir(parents=True)
        fifo = self.root / "fifo.json"
        os.mkfifo(fifo)
        started = time.monotonic()
        self.assertEqual(state.read_json(fifo, "safe"), "safe")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_existing_state_directory_permissions_are_tightened(self):
        self.root.mkdir(mode=0o775)
        os.chmod(self.root, 0o775)
        state.save_state({"ok": True})
        self.assertEqual(file_mode(self.root), 0o700)

    def test_write_json_refuses_symlinked_files_roots_and_outside_paths(self):
        outside = self.home / "outside.json"
        outside.write_text('{"keep":true}', encoding="utf-8")
        self.root.mkdir(parents=True, exist_ok=True)
        link = self.root / "state.json"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            state.write_json(link, {"replace": True})
        self.assertEqual(json.loads(outside.read_text(encoding="utf-8")), {"keep": True})
        with self.assertRaises(ValueError):
            state.write_json(self.home / "outside-state.json", {})

    def test_failed_write_leaves_previous_file_and_no_temp(self):
        target = self.root / "data.json"
        state.write_json(target, {"keep": True})
        loop: list = []
        loop.append(loop)
        with self.assertRaises(ValueError):
            state.write_json(target, loop)
        self.assertEqual(state.read_json(target), {"keep": True})
        self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ["data.json"])

    def test_refuses_core_owned_files(self):
        for name in (".usage.json", ".curator_state"):
            with self.assertRaises(ValueError):
                state.write_json(self.home / "skills" / name, {"x": 1})


class StateStoreTests(StateTestCase):
    def test_state_roundtrip(self):
        self.assertEqual(state.load_state(), {})
        state.save_state({"last_run": {"mode": "observe"}, "seen": ["a", "b"]})
        self.assertEqual(state.load_state()["seen"], ["a", "b"])
        self.assertEqual(file_mode(state.state_path()), 0o600)


class RelationCacheTests(StateTestCase):
    def test_key_is_content_hash(self):
        base = state.relation_key("d1", "d2", contract_version="v1")
        self.assertEqual(base, state.relation_key("d2", "d1", contract_version="v1"))
        self.assertNotEqual(base, state.relation_key("d1", "d3", contract_version="v1"))
        self.assertNotEqual(base, state.relation_key("d1", "d2", contract_version="v2"))
        self.assertNotEqual(base, state.relation_key("d1", "d2", contract_version="v1", model="m"))
        self.assertRegex(base, r"^[0-9a-f]{32}$")

    def test_roundtrip_and_invalid_entries(self):
        judgment = make_judgment()
        cache = {}
        key = state.remember_relation(cache, judgment, model="jev-latest")
        self.assertEqual(key, state.relation_key(judgment.a_digest, judgment.b_digest,
                                                 contract_version=judgment.contract_version,
                                                 model="jev-latest"))
        self.assertEqual(state.cached_relation(cache, "alpha-d1", "beta-d2",
                                               contract_version="skill-relations-v1",
                                               model="jev-latest"), judgment)
        self.assertIsNone(state.cached_relation(cache, "alpha-d1", "beta-d2",
                                                contract_version="skill-relations-v1"))
        state.save_relation_cache(cache)
        loaded = state.load_relation_cache()
        self.assertEqual(state.cached_relation(loaded, "alpha-d1", "beta-d2",
                                               contract_version="skill-relations-v1",
                                               model="jev-latest"), judgment)
        self.assertIsNone(state.cached_relation(loaded, "alpha-d1", "beta-d2",
                                                contract_version="skill-relations-v1", model="other"))
        self.assertIsNone(state.cached_relation({"k": {"judgment": {"relation": "made-up"}}},
                                                "x", "y", contract_version="v"))
        self.assertIsNone(state.cached_relation({"k": "junk"}, "x", "y", contract_version="v"))
        malformed = state.judgment_to_dict(judgment)
        malformed["probabilities"] = {"same_class": "0.8"}
        self.assertIsNone(state.judgment_from_dict(malformed))

    def test_eviction_keeps_newest(self):
        cache = {}
        keys = [state.remember_relation(cache, make_judgment(a=f"s{i}", b="base"), model="m")
                for i in range(3)]
        state.remember_relation(cache, make_judgment(a="s0", b="base"), model="m")  # refresh s0
        state.save_relation_cache(cache, max_entries=2)
        self.assertEqual(set(state.load_relation_cache()), {keys[0], keys[2]})


class LockTests(StateTestCase):
    def test_exclusive_claim_release_and_nested_busy(self):
        with state.claim_lock() as held:
            self.assertTrue(held)
            lock = state.lock_path()
            self.assertTrue(lock.is_file())
            self.assertEqual(file_mode(lock), 0o600)
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["pid"], os.getpid())
            with state.claim_lock() as nested:
                self.assertFalse(nested)
        self.assertFalse(state.lock_path().exists())

    def test_stale_recovery_dead_pid_and_age(self):
        lock = state.lock_path()
        lock.parent.mkdir(parents=True, exist_ok=True)
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        dead_pid = child.pid
        child.wait()
        lock.write_text(json.dumps({"pid": dead_pid, "at": 0}), encoding="utf-8")
        with state.claim_lock(stale_seconds=3600) as held:
            self.assertTrue(held)
        lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        with state.claim_lock() as held:
            self.assertFalse(held)  # live owner, fresh claim
        old = time.time() - 7200
        os.utime(lock, (old, old))
        with state.claim_lock(stale_seconds=1800) as held:
            self.assertFalse(held)  # age never steals a lock from a live owner
        lock.write_text("garbage", encoding="utf-8")
        with state.claim_lock() as held:
            self.assertFalse(held)  # unparseable claim respected while fresh
        os.utime(lock, (old, old))
        with state.claim_lock() as held:
            self.assertTrue(held)

    def test_concurrent_stale_recovery_has_one_owner(self):
        lock = state.lock_path()
        lock.parent.mkdir(parents=True, exist_ok=True)
        worker = """
import os, sys, time
from pathlib import Path
os.environ['HERMES_HOME'] = sys.argv[1]
sys.path.insert(0, sys.argv[2])
from plugin import state
barrier, release = Path(sys.argv[3]), Path(sys.argv[4])
while not barrier.exists():
    time.sleep(0.0005)
with state.claim_lock() as held:
    print('HELD' if held else 'BUSY', flush=True)
    while held and not release.exists():
        time.sleep(0.0005)
"""
        repo = str(Path(__file__).resolve().parents[2])
        for trial in range(12):
            lock.write_text(json.dumps({"pid": 999999, "at": 0}), encoding="utf-8")
            old = time.time() - 7200
            os.utime(lock, (old, old))
            barrier = self.home / f"barrier-{trial}"
            release = self.home / f"release-{trial}"
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", worker, str(self.home), repo, str(barrier), str(release)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            barrier.touch()
            outputs = []
            for process in processes:
                self.assertIsNotNone(process.stdout)
                outputs.append(process.stdout.readline().strip())  # type: ignore[union-attr]
            self.assertEqual(outputs.count("HELD"), 1, outputs)
            release.touch()
            for process in processes:
                _, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stderr)
            lock.unlink(missing_ok=True)

    @unittest.skipUnless(hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK"),
                         "requires POSIX nonblocking FIFOs")
    def test_lock_metadata_reads_are_bounded_nonblocking_and_nofollow(self):
        self.root.mkdir(parents=True)
        lock = state.lock_path()
        os.mkfifo(lock)
        started = time.monotonic()
        with state.claim_lock() as held:
            self.assertFalse(held)
        self.assertLess(time.monotonic() - started, 1.0)

        lock.unlink()
        lock.write_bytes(b"x" * 8192)
        self.assertIsNone(state._lock_pid(lock))

        victim = self.home / "victim.json"
        victim.write_text(json.dumps({"pid": 999999}), encoding="utf-8")
        lock.unlink()
        lock.symlink_to(victim)
        self.assertIsNone(state._lock_pid(lock))

    def test_claim_lock_yields_false_for_symlinked_state_root(self):
        outside = self.home / "outside"
        outside.mkdir()
        self.root.symlink_to(outside, target_is_directory=True)
        with state.claim_lock() as held:
            self.assertFalse(held)
        self.assertEqual(list(outside.iterdir()), [])


class ReportTests(StateTestCase):
    def test_reports_roundtrip_and_name_safety(self):
        path = state.write_report("run-1", {"status": "observe", "pairs": 3})
        self.assertEqual(path, state.reports_dir() / "run-1.json")
        self.assertEqual(file_mode(path), 0o600)
        self.assertEqual((state.read_report("run-1") or {}).get("pairs"), 3)
        self.assertEqual(state.list_reports(), ["run-1"])
        self.assertIsNone(state.read_report("missing"))
        plan = MergePlan(plan_id="p1", canonical="a", canonical_digest="d", absorbed=("b",),
                         absorbed_digests={"b": "d2"}, relation_keys=("a::b",))
        state.write_report("plan-1", plan)
        self.assertEqual((state.read_report("plan-1") or {}).get("absorbed"), ["b"])
        outside = self.home / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        (state.reports_dir() / "evil.json").symlink_to(outside)
        self.assertEqual(state.list_reports(), ["plan-1", "run-1"])
        for bad in ("", "../escape", "a/b", ".hidden", "x" * 65, "a..b"):
            with self.assertRaises(ValueError):
                state.write_report(bad, {})
            with self.assertRaises(ValueError):
                state.read_report(bad)


class CoreBoundaryTests(StateTestCase):
    def test_core_files_are_never_touched(self):
        fake = self.home / "hermes"
        skills = fake / "skills"
        skills.mkdir(parents=True)
        core = {skills / ".usage.json": b'{"core": "usage"}',
                skills / ".curator_state": b'{"core": "curator"}'}
        for path, data in core.items():
            path.write_bytes(data)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in core}
        os.environ["HERMES_HOME"] = str(fake)
        self.assertEqual(state.state_root(), fake / "jev-curator")
        state.audit("tick")
        state.save_state({"ok": True})
        cache = {}
        state.remember_relation(cache, make_judgment())
        state.save_relation_cache(cache)
        state.write_report("run-1", {"plan": "none"})
        with state.claim_lock() as held:
            self.assertTrue(held)
        for path, (data, mtime) in before.items():
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
        self.assertTrue((fake / "jev-curator" / "audit.jsonl").is_file())
        self.assertFalse((skills / "jev-curator").exists())


if __name__ == "__main__":
    unittest.main()
