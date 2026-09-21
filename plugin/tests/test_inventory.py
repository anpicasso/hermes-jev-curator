"""Adversarial tests for plugin.inventory.

Every test runs against throwaway skill roots and stubbed Hermes internals
(agent.skill_utils / hermes_constants / tools.skill_usage / cron.jobs), so no
test can read or write the real ~/.hermes profile, usage store, or cron store.
The stubs mirror the real module shapes (including the real walker's
``followlinks=True``), which is what makes the escape tests meaningful.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugin import inventory  # noqa: E402
from plugin.models import SkillArtifact  # noqa: E402


_MISSING = object()

# Mirrors agent.skill_utils so the stub walker prunes what the real one prunes.
_SUPPORT_DIRS = frozenset(("references", "templates", "assets", "scripts"))
_EXCLUDED_DIRS = frozenset((
    ".git", ".github", ".hub", ".archive", ".curator_backups", ".locks",
    ".venv", "venv", "node_modules", "site-packages", "__pycache__",
    ".tox", ".nox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
))

_POSIX_ONLY = unittest.skipIf(
    os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() == 0,
    "file modes and FIFOs need a non-root POSIX user",
)


def row(name, *, provenance="agent", pinned=False, state="active",
        use_count=0, last_activity_at="") -> dict[str, Any]:
    data: dict[str, Any] = {"name": name, "pinned": pinned, "state": state,
                            "use_count": use_count, "last_activity_at": last_activity_at}
    if provenance is not _MISSING:
        data["provenance"] = provenance
    return data


def write(path: Path, content) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def skill_doc(name=None, description="", body="Body.\n") -> str:
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description:
        lines.append(f"description: {description}")
    lines += ["---", body]
    return "\n".join(lines)


def expected_digest(package: Path, omit_data=frozenset()) -> str:
    """Independent reimplementation of the documented package-hash contract:
    every non-symlink file contributes ``relative\\0size\\0`` plus its bytes."""
    hasher = hashlib.sha256()
    for path in sorted(package.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        hasher.update(relative.encode("utf-8", "surrogateescape") + b"\0")
        hasher.update(str(path.lstat().st_size).encode() + b"\0")
        if relative not in omit_data:
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _walk_skill_index_files(skills_dir, filename):
    """Mirrors agent.skill_utils.iter_skill_index_files: os.walk with
    followlinks=True, pruning excluded dirs and support dirs of skill roots."""
    matches = []
    for dirpath, dirnames, filenames in os.walk(str(skills_dir), followlinks=True):
        has_index = filename in filenames
        dirnames[:] = [name for name in dirnames
                       if name not in _EXCLUDED_DIRS and not (has_index and name in _SUPPORT_DIRS)]
        if has_index:
            matches.append(os.path.join(dirpath, filename))
    yield from map(Path, sorted(matches))


def _parse_frontmatter(content: str):
    """Small stand-in for the real YAML frontmatter parser."""
    content = content.removeprefix("\ufeff")
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    block = content[3:end]
    body = content[end + 4:].lstrip("\n")
    frontmatter: dict[str, Any] = {}
    for line in block.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            raw = value.strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
                value = raw[1:-1]
            else:
                value = raw
            frontmatter[key.strip()] = value
    return frontmatter, body


class HermesStubs:
    """Fake Hermes modules installed in sys.modules for the duration of a test.

    inventory imports these lazily inside collect_inventory, so swapping the
    sys.modules entries is enough to keep the real profile untouched.
    """

    _MODULES = ("agent", "agent.skill_utils", "hermes_constants", "tools",
                "tools.skill_usage", "cron", "cron.jobs")

    def __init__(self, *, home, curated=(), unmanaged=None, provenance="agent",
                 provenance_error=None, eligible=True, eligibility_error=None,
                 cron_names=(), cron_error=None, disabled_names=(),
                 disabled_error=None, essential_names=("hermes-agent",),
                 org_mirror_names=()):
        self.home = Path(home)
        self.curated = [dict(item) for item in curated]
        self.unmanaged = [dict(item) for item in (curated if unmanaged is None else unmanaged)]
        self.provenance_value = provenance
        self.provenance_error = provenance_error
        self.eligible = eligible
        self.eligibility_error = eligibility_error
        self.cron_names = set(cron_names)
        self.cron_error = cron_error
        self.disabled_names = set(disabled_names)
        self.disabled_error = disabled_error
        self.essential_names = frozenset(essential_names)
        self.org_mirror_names = set(org_mirror_names)
        self.calls: list[str] = []
        self.parse_inputs: list[str] = []
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> "HermesStubs":
        agent = types.ModuleType("agent")
        agent.__path__ = []
        skill_utils = types.ModuleType("agent.skill_utils")

        def parse_frontmatter(content):
            self.parse_inputs.append(content)
            return _parse_frontmatter(content)

        def get_disabled_skill_names(platform=None):
            self.calls.append("get_disabled_skill_names")
            if self.disabled_error is not None:
                raise self.disabled_error
            return set(self.disabled_names)

        def is_org_mirror_path(path, skills_dir):
            self.calls.append("is_org_mirror_path")
            return Path(path).name in self.org_mirror_names

        setattr(skill_utils, "iter_skill_index_files", _walk_skill_index_files)
        setattr(skill_utils, "parse_frontmatter", parse_frontmatter)
        setattr(skill_utils, "get_disabled_skill_names", get_disabled_skill_names)
        setattr(skill_utils, "is_org_mirror_path", is_org_mirror_path)
        setattr(skill_utils, "ESSENTIAL_SKILLS", self.essential_names)
        setattr(agent, "skill_utils", skill_utils)

        constants = types.ModuleType("hermes_constants")

        def get_hermes_home():
            self.calls.append("get_hermes_home")
            return self.home

        setattr(constants, "get_hermes_home", get_hermes_home)

        tools = types.ModuleType("tools")
        tools.__path__ = []
        usage = types.ModuleType("tools.skill_usage")

        def curated_report():
            self.calls.append("curated_report")
            return [dict(item) for item in self.curated]

        def usage_report():
            self.calls.append("usage_report")
            return [dict(item) for item in self.unmanaged]

        def provenance(name):
            self.calls.append(f"provenance:{name}")
            if self.provenance_error is not None:
                raise self.provenance_error
            return self.provenance_value

        def is_curation_eligible(name, package=None):
            self.calls.append(f"eligible:{name}")
            if self.eligibility_error is not None:
                raise self.eligibility_error
            return self.eligible

        setattr(usage, "curated_report", curated_report)
        setattr(usage, "usage_report", usage_report)
        setattr(usage, "provenance", provenance)
        setattr(usage, "is_curation_eligible", is_curation_eligible)
        setattr(tools, "skill_usage", usage)

        cron = types.ModuleType("cron")
        cron.__path__ = []
        jobs = types.ModuleType("cron.jobs")

        def referenced_skill_names():
            self.calls.append("referenced_skill_names")
            if self.cron_error is not None:
                raise self.cron_error
            return set(self.cron_names)

        setattr(jobs, "referenced_skill_names", referenced_skill_names)
        setattr(cron, "jobs", jobs)

        modules = {"agent": agent, "agent.skill_utils": skill_utils,
                   "hermes_constants": constants, "tools": tools,
                   "tools.skill_usage": usage, "cron": cron, "cron.jobs": jobs}
        for name in self._MODULES:
            self._saved[name] = sys.modules.get(name, _MISSING)
            sys.modules[name] = modules[name]
        return self

    def __exit__(self, *exc):
        for name in self._MODULES:
            saved = self._saved[name]
            if saved is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
        return False


class InventoryTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.root = self.tmp / "skills"
        self.root.mkdir()
        self.home = self.tmp / "home"
        (self.home / "skills").mkdir(parents=True)

    @contextlib.contextmanager
    def hermes(self, **kwargs):
        kwargs.setdefault("home", self.home)
        with HermesStubs(**kwargs) as stubs:
            yield stubs

    def make_skill(self, dirname, name=None, *, body="Body.\n", description="", extra=None):
        package = self.root / dirname
        write(package / "SKILL.md", skill_doc(name, description, body))
        for relative, content in (extra or {}).items():
            write(package / relative, content)
        return package

    def collect(self, **kwargs):
        return inventory.collect_inventory(skills_root=self.root, **kwargs)


class RootAndRoutingTests(InventoryTestCase):
    def test_missing_root_returns_empty(self):
        with self.hermes(unmanaged=[row("anything")]):
            artifacts = inventory.collect_inventory(
                skills_root=self.tmp / "nope", include_unmanaged=True)
        self.assertEqual(artifacts, [])

    def test_default_root_comes_from_stubbed_hermes_home(self):
        write(self.home / "skills" / "home-skill" / "SKILL.md", skill_doc("home-skill"))
        with self.hermes(curated=[row("home-skill")]) as stubs:
            artifacts = inventory.collect_inventory()
        self.assertEqual([item.name for item in artifacts], ["home-skill"])
        self.assertIn("get_hermes_home", stubs.calls)

    def test_curated_report_used_by_default(self):
        self.make_skill("managed", "managed")
        self.make_skill("extra", "extra")
        with self.hermes(curated=[row("managed")],
                         unmanaged=[row("managed"), row("extra")]) as stubs:
            artifacts = self.collect()
        self.assertEqual([item.name for item in artifacts], ["managed"])
        self.assertIn("curated_report", stubs.calls)
        self.assertNotIn("usage_report", stubs.calls)

    def test_include_unmanaged_uses_usage_report(self):
        self.make_skill("managed", "managed")
        self.make_skill("extra", "extra")
        with self.hermes(curated=[row("managed")],
                         unmanaged=[row("managed"), row("extra")]) as stubs:
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual([item.name for item in artifacts], ["extra", "managed"])
        self.assertIn("usage_report", stubs.calls)
        self.assertNotIn("curated_report", stubs.calls)

    def test_name_absent_from_report_is_skipped(self):
        self.make_skill("ghost", "ghost")
        with self.hermes(unmanaged=[row("other")]):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts, [])

    def test_duplicate_names_deduped_in_walk_order(self):
        self.make_skill("a-first", "same", body="FIRST_BODY")
        self.make_skill("z-second", "same", body="SECOND_BODY")
        with self.hermes(curated=[row("same")]):
            artifacts = self.collect()
        self.assertEqual(len(artifacts), 1)
        self.assertIn("FIRST_BODY", artifacts[0].text)
        self.assertNotIn("SECOND_BODY", artifacts[0].text)

    def test_name_falls_back_to_directory_name(self):
        self.make_skill("dir-named", None)
        with self.hermes(curated=[row("dir-named")]):
            artifacts = self.collect()
        self.assertEqual([item.name for item in artifacts], ["dir-named"])

    def test_blank_frontmatter_name_skipped(self):
        self.make_skill("blank", '"   "')
        with self.hermes(unmanaged=[row("blank")]):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts, [])

    def test_malformed_frontmatter_falls_back_to_directory_name(self):
        package = self.root / "malformed"
        write(package / "SKILL.md", "---\nname: never-closed\nbody without fence\n")
        with self.hermes(unmanaged=[row("malformed")]):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual([item.name for item in artifacts], ["malformed"])

    def test_row_metadata_defaults_and_safe_use_count(self):
        self.make_skill("weird", "weird")
        self.make_skill("neg", "neg")
        self.make_skill("junk", "junk")
        rows = [
            row("weird", use_count="7", state=None, last_activity_at=None, pinned=1),
            row("neg", use_count=-3),
            row("junk", use_count="not-a-number"),
        ]
        with self.hermes(curated=rows):
            artifacts = self.collect()
        by_name = {item.name: item for item in artifacts}
        self.assertEqual(by_name["weird"].use_count, 7)
        self.assertEqual(by_name["weird"].state, "active")
        self.assertEqual(by_name["weird"].last_activity_at, "")
        self.assertTrue(by_name["weird"].pinned)
        self.assertEqual(by_name["neg"].use_count, 0)
        self.assertEqual(by_name["junk"].use_count, 0)

    def test_artifact_by_name_reads_unmanaged_and_returns_none_for_unknown(self):
        self.make_skill("only-unmanaged", "only-unmanaged")
        with self.hermes(curated=[], unmanaged=[row("only-unmanaged")]) as stubs:
            found = inventory.artifact_by_name("only-unmanaged", skills_root=self.root)
            missing = inventory.artifact_by_name("nope", skills_root=self.root)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.name, "only-unmanaged")
        self.assertIsNone(missing)
        self.assertIn("usage_report", stubs.calls)

    def test_names_filter_hashes_only_requested_packages(self):
        self.make_skill("one", "one")
        self.make_skill("two", "two")
        with self.hermes(unmanaged=[row("one"), row("two")]):
            with mock.patch.object(inventory, "_read_package", wraps=inventory._read_package) as read:
                artifacts = self.collect(include_unmanaged=True, names={"one"})
        self.assertEqual([item.name for item in artifacts], ["one"])
        self.assertEqual([call.args[0].name for call in read.call_args_list], ["one"])

    def test_path_alias_filter_finds_and_protects_name_mismatch(self):
        package = self.make_skill("directory-name", "frontmatter-name")
        with self.hermes(unmanaged=[row("directory-name")]):
            artifacts = self.collect(include_unmanaged=True, names={"directory-name"})
        self.assertEqual([item.name for item in artifacts], ["frontmatter-name"])
        self.assertEqual(artifacts[0].path, package.resolve())
        self.assertIn("name-path-mismatch", artifacts[0].protected_reasons)
        self.assertIn("directory-name", inventory.artifact_host_names(
            artifacts[0].name, artifacts[0].path, skills_root=self.root))

    def test_categorized_path_is_a_host_alias(self):
        package = self.make_skill("mlops/axolotl", "axolotl")
        with self.hermes(unmanaged=[row("axolotl")]):
            artifacts = self.collect(include_unmanaged=True, names={"mlops/axolotl"})
        self.assertEqual([item.name for item in artifacts], ["axolotl"])
        self.assertIn("mlops/axolotl", inventory.artifact_host_names(
            artifacts[0].name, package, skills_root=self.root))


class SymlinkAndEscapeTests(InventoryTestCase):
    def test_symlinked_skill_md_never_read(self):
        secret = write(self.tmp / "outside" / "SKILL.md",
                       skill_doc("sneak", body="TOPSECRET_MARKER"))
        package = self.root / "sneak"
        package.mkdir()
        os.symlink(secret, package / "SKILL.md")
        ghost = self.root / "ghost"
        ghost.mkdir()
        os.symlink(self.tmp / "does-not-exist", ghost / "SKILL.md")
        with self.hermes(unmanaged=[row("sneak"), row("ghost")]) as stubs:
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts, [])
        self.assertFalse(any("TOPSECRET" in raw for raw in stubs.parse_inputs))

    def test_package_behind_symlinked_dir_never_read(self):
        outside = self.tmp / "outside-pkg"
        write(outside / "SKILL.md", skill_doc("sneak", body="TOPSECRET_MARKER"))
        write(outside / "references" / "leak.md", "TOPSECRET_MARKER")
        os.symlink(outside, self.root / "sneak")
        with self.hermes(unmanaged=[row("sneak")]) as stubs:
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts, [])
        self.assertFalse(any("TOPSECRET" in raw for raw in stubs.parse_inputs))

    def test_in_root_alias_symlink_deduped(self):
        real = self.make_skill("real", "real", body="REAL_BODY")
        os.symlink(real, self.root / "alias")
        with self.hermes(curated=[row("real")]):
            artifacts = self.collect()
        self.assertEqual([item.name for item in artifacts], ["real"])
        self.assertIn("REAL_BODY", artifacts[0].text)
        self.assertEqual(artifacts[0].digest, inventory._read_package(real)[2])

    def test_symlink_inside_package_flagged_and_not_followed(self):
        secret = write(self.tmp / "outside-secret.md", "TOPSECRET_MARKER")
        package = self.make_skill("leaky", "leaky")
        (package / "references").mkdir()
        os.symlink(secret, package / "references" / "leak.md")
        os.symlink(secret, package / "top-secret.md")
        with self.hermes(unmanaged=[row("leaky")]):
            artifacts = self.collect(include_unmanaged=True)
        artifact = artifacts[0]
        self.assertTrue(artifact.protected)
        self.assertIn("contains-symlink", artifact.protected_reasons)
        self.assertNotIn("TOPSECRET", artifact.text)
        self.assertNotIn("references/leak.md", artifact.support_files)
        self.assertNotIn("top-secret.md", artifact.support_files)
        self.assertEqual(artifact.digest, expected_digest(package))

    def test_dangling_symlink_inside_package_flagged_without_error(self):
        package = self.make_skill("dangling", "dangling")
        (package / "scripts").mkdir()
        os.symlink(self.tmp / "missing-target", package / "scripts" / "dead.sh")
        text, support, digest, flags = inventory._read_package(package)
        self.assertIn("contains-symlink", flags)
        self.assertNotIn("scripts/dead.sh", support)
        self.assertEqual(digest, expected_digest(package))

    @_POSIX_ONLY
    def test_fifo_never_opened(self):
        package = self.make_skill("pipes", "pipes")
        (package / "assets").mkdir()
        os.mkfifo(package / "assets" / "pipe")
        with self.hermes(unmanaged=[row("pipes")]):
            artifacts = self.collect(include_unmanaged=True)
        artifact = artifacts[0]
        self.assertNotIn("assets/pipe", artifact.support_files)
        self.assertNotIn("assets/pipe", artifact.text)
        self.assertNotIn("unreadable-entry", artifact.protected_reasons)

    @_POSIX_ONLY
    def test_unreadable_file_flagged_not_read(self):
        package = self.make_skill("locked", "locked")
        locked = write(package / "references" / "locked.md", "LOCKED_CONTENT")
        os.chmod(locked, 0)
        self.addCleanup(os.chmod, locked, 0o644)
        with self.hermes(unmanaged=[row("locked")]):
            artifacts = self.collect(include_unmanaged=True)
        artifact = artifacts[0]
        self.assertIn("unreadable-entry", artifact.protected_reasons)
        self.assertNotIn("LOCKED_CONTENT", artifact.text)
        self.assertNotIn("references/locked.md", artifact.support_files)
        self.assertEqual(artifact.digest, expected_digest(
            package, omit_data={"references/locked.md"}))

    @_POSIX_ONLY
    def test_unreadable_subdir_contents_invisible(self):
        package = self.make_skill("dark", "dark")
        hidden = package / "references" / "sub"
        write(hidden / "hidden.md", "HIDDEN_CONTENT")
        os.chmod(hidden, 0)
        self.addCleanup(os.chmod, hidden, 0o755)
        with self.hermes(unmanaged=[row("dark")]):
            artifacts = self.collect(include_unmanaged=True)
        artifact = artifacts[0]
        self.assertNotIn("HIDDEN_CONTENT", artifact.text)
        self.assertNotIn("references/sub/hidden.md", artifact.support_files)
        self.assertEqual(artifact.protected_reasons, ())


class TextProtectionTests(InventoryTestCase):
    def test_support_files_collected_and_text_headed(self):
        package = self.make_skill("supported", "supported", extra={
            "references/ref.md": "REF_CONTENT",
            "references/sub/deep.md": "DEEP_CONTENT",
            "templates/tpl.md": "TPL_CONTENT",
            "scripts/run.py": "print('RUN_CONTENT')",
            "assets/pic.css": "CSS_CONTENT",
            "assets/blob.bin": b"\x00\x01BIN_CONTENT",
        })
        text, support, digest, flags = inventory._read_package(package)
        self.assertEqual(support, ["assets/blob.bin", "assets/pic.css", "references/ref.md",
                                   "references/sub/deep.md", "scripts/run.py", "templates/tpl.md"])
        self.assertIn("===== references/ref.md =====", text)
        self.assertIn("REF_CONTENT", text)
        self.assertIn("DEEP_CONTENT", text)
        self.assertIn("RUN_CONTENT", text)
        self.assertIn("CSS_CONTENT", text)
        self.assertNotIn("BIN_CONTENT", text)
        self.assertEqual(flags, ("contains-binary",))
        self.assertEqual(digest, expected_digest(package))

    def test_non_support_files_hashed_but_not_exposed(self):
        package = self.make_skill("junk", "junk", extra={"notes.txt": "JUNK_CONTENT"})
        text, support, digest, flags = inventory._read_package(package)
        self.assertNotIn("JUNK_CONTENT", text)
        self.assertEqual(support, [])
        self.assertEqual(digest, expected_digest(package))

    def test_oversized_text_file_flagged_and_omitted(self):
        package = self.make_skill("big", "big", extra={
            "references/huge.md": "HUGE_MARKER" + "A" * 300_000,
        })
        with self.hermes(unmanaged=[row("big")]):
            artifacts = self.collect(include_unmanaged=True)
        artifact = artifacts[0]
        self.assertIn("text-too-large", artifact.protected_reasons)
        self.assertNotIn("HUGE_MARKER", artifact.text)
        self.assertLess(len(artifact.text), 100_000)
        self.assertIn("references/huge.md", artifact.support_files)

    def test_package_text_budget_flagged(self):
        package = self.make_skill("budget", "budget", extra={
            "references/a.md": "BUDGET_1 " + "a" * 691,
            "references/b.md": "BUDGET_2 " + "b" * 691,
            "references/c.md": "BUDGET_3 " + "c" * 691,
        })
        with mock.patch.object(inventory, "_MAX_TEXT_FILE_BYTES", 1_000), \
                mock.patch.object(inventory, "_MAX_PACKAGE_TEXT_BYTES", 1_500):
            text, support, digest, flags = inventory._read_package(package)
        self.assertIn("text-too-large", flags)
        self.assertIn("BUDGET_1", text)
        self.assertIn("BUDGET_2", text)
        self.assertNotIn("BUDGET_3", text)

    def test_binary_file_never_decoded(self):
        package = self.make_skill("binary", "binary", extra={
            "references/blob.md": b"\x00\x01BINARY_MARKER" + b"\x00" * 64,
        })
        text, support, digest, flags = inventory._read_package(package)
        self.assertEqual(flags, ("contains-binary",))
        self.assertNotIn("BINARY_MARKER", text)
        self.assertIn("references/blob.md", support)
        self.assertEqual(digest, expected_digest(package))

    def test_non_utf8_file_flagged_not_decoded(self):
        package = self.make_skill("latin", "latin", extra={
            "references/bad.md": b"caf\xe9 NONUTF8_MARKER",
        })
        text, support, digest, flags = inventory._read_package(package)
        self.assertIn("contains-non-utf8", flags)
        self.assertNotIn("NONUTF8_MARKER", text)
        self.assertIn("references/bad.md", support)
        self.assertEqual(digest, expected_digest(package))

    def test_non_utf8_skill_md_dropped_silently(self):
        package = self.root / "broken"
        package.mkdir()
        (package / "SKILL.md").write_bytes(b"---\nname: broken\n---\n\xff\xfe\xff")
        with self.hermes(unmanaged=[row("broken")]):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts, [])

    def test_oversized_skill_md_flagged_at_collect(self):
        self.make_skill("giant", "giant", body="A" * 300_000)
        with self.hermes(unmanaged=[row("giant")]):
            artifacts = self.collect(include_unmanaged=True)
        artifact = artifacts[0]
        self.assertIn("text-too-large", artifact.protected_reasons)
        self.assertEqual(artifact.text, "")

    def test_binary_asset_flags_package_at_collect(self):
        self.make_skill("biny", "biny", extra={"assets/logo.png": b"\x00\x01BINARY"})
        with self.hermes(unmanaged=[row("biny")]):
            artifacts = self.collect(include_unmanaged=True)
        artifact = artifacts[0]
        self.assertIn("contains-binary", artifact.protected_reasons)
        self.assertTrue(artifact.protected)


class DigestTests(InventoryTestCase):
    def test_digest_is_sha256_and_stable(self):
        package = self.make_skill("stable", "stable", extra={"references/a.md": "A"})
        first = inventory._read_package(package)[2]
        second = inventory._read_package(package)[2]
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(first, expected_digest(package))

    def test_digest_tracks_skill_support_binary_and_junk_content(self):
        package = self.make_skill("tracked", "tracked", extra={
            "references/ref.md": "REF_V1",
            "references/blob.bin": b"\x00BIN_V1",
            "notes.txt": "JUNK_V1",
        })
        digests = [inventory._read_package(package)[2]]
        for target, content in ((package / "SKILL.md", skill_doc("tracked", body="Body v2")),
                                (package / "references" / "ref.md", "REF_V2"),
                                (package / "references" / "blob.bin", b"\x00BIN_V2"),
                                (package / "notes.txt", "JUNK_V2")):
            write(target, content)
            digests.append(inventory._read_package(package)[2])
        self.assertEqual(len(set(digests)), len(digests), "every content change must move the digest")

    def test_digest_changes_when_path_changes(self):
        package = self.make_skill("renamed", "renamed", extra={"references/one.md": "SAME"})
        before = inventory._read_package(package)[2]
        (package / "references" / "one.md").rename(package / "references" / "two.md")
        after = inventory._read_package(package)[2]
        self.assertNotEqual(before, after)
        self.assertEqual(after, expected_digest(package))

    def test_symlink_presence_flagged_but_not_hashed(self):
        package = self.make_skill("flagged", "flagged")
        before = inventory._read_package(package)[2]
        os.symlink(self.tmp / "outside-secret.md", package / "leak.md")
        text, support, after, flags = inventory._read_package(package)
        self.assertIn("contains-symlink", flags)
        self.assertEqual(before, after)
        self.assertNotIn("leak.md", support)

    def test_package_hash_cap_skips_data_but_keeps_path_and_size(self):
        package = self.tmp / "capped"
        write(package / "a.bin", b"a" * 40)
        write(package / "b.bin", b"b" * 40)
        with mock.patch.object(inventory, "_MAX_HASH_BYTES", 70):
            text, support, digest, flags = inventory._read_package(package)
            self.assertIn("package-too-large", flags)
            self.assertEqual(digest, expected_digest(package, omit_data={"b.bin"}))
            write(package / "b.bin", b"c" * 40)
            self.assertEqual(inventory._read_package(package)[2], digest,
                             "skipped data must not enter the hash")
            write(package / "a.bin", b"d" * 40)
            self.assertNotEqual(inventory._read_package(package)[2], digest,
                                "hashed data must enter the hash")


class ProtectionTests(InventoryTestCase):
    def test_frontmatter_name_directory_mismatch_is_protected(self):
        self.make_skill("directory-name", "frontmatter-name")
        with self.hermes(curated=[row("frontmatter-name")]):
            artifacts = self.collect()
        self.assertEqual([item.name for item in artifacts], ["frontmatter-name"])
        self.assertIn("name-path-mismatch", artifacts[0].protected_reasons)

    def test_provenance_bundled_hub_external_flagged(self):
        for name, provenance in (("bun", "bundled"), ("hub", "hub"),
                                 ("ext", "external"), ("mine", "agent")):
            self.make_skill(name, name)
        rows = [row("bun", provenance="bundled"), row("hub", provenance="hub"),
                row("ext", provenance="external"), row("mine", provenance="agent")]
        with self.hermes(unmanaged=rows):
            artifacts = self.collect(include_unmanaged=True)
        by_name = {item.name: item for item in artifacts}
        self.assertEqual(by_name["bun"].protected_reasons, ("bundled",))
        self.assertEqual(by_name["hub"].protected_reasons, ("hub",))
        self.assertEqual(by_name["ext"].protected_reasons, ("external",))
        self.assertEqual(by_name["mine"].protected_reasons, ())
        self.assertEqual(by_name["mine"].provenance, "agent")
        self.assertFalse(by_name["mine"].protected)

    def test_provenance_falls_back_to_usage_module(self):
        self.make_skill("fallback", "fallback")
        with self.hermes(unmanaged=[row("fallback", provenance=_MISSING)],
                         provenance="hub") as stubs:
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts[0].protected_reasons, ("hub",))
        self.assertIn("provenance:fallback", stubs.calls)

    def test_provenance_lookup_error_propagates_fail_closed(self):
        self.make_skill("fallback", "fallback")
        with self.hermes(unmanaged=[row("fallback", provenance=_MISSING)],
                         provenance_error=RuntimeError("store unavailable")):
            with self.assertRaises(RuntimeError):
                self.collect(include_unmanaged=True)

    def test_pinned_flag(self):
        self.make_skill("pin", "pin")
        with self.hermes(unmanaged=[row("pin", pinned=True)]):
            artifacts = self.collect(include_unmanaged=True)
        self.assertTrue(artifacts[0].pinned)
        self.assertEqual(artifacts[0].protected_reasons, ("pinned",))

    def test_disabled_skill_flagged(self):
        self.make_skill("off", "off")
        with self.hermes(unmanaged=[row("off")], disabled_names={"off"}) as stubs:
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts[0].protected_reasons, ("disabled",))
        self.assertIn("get_disabled_skill_names", stubs.calls)

    def test_essential_skill_flagged(self):
        self.make_skill("hermes-agent", "hermes-agent")
        with self.hermes(unmanaged=[row("hermes-agent")]):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts[0].protected_reasons, ("essential",))

    def test_disabled_lookup_error_propagates_fail_closed(self):
        # Unlike the cron and eligibility lookups, get_disabled_skill_names is
        # not wrapped: a config-read failure aborts the whole inventory.
        self.make_skill("off", "off")
        with self.hermes(unmanaged=[row("off")],
                         disabled_error=RuntimeError("config unreadable")):
            with self.assertRaises(RuntimeError):
                self.collect(include_unmanaged=True)

    def test_org_mirror_skill_flagged(self):
        self.make_skill("mirrored", "mirrored")
        with self.hermes(unmanaged=[row("mirrored")],
                         org_mirror_names={"mirrored"}) as stubs:
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts[0].protected_reasons, ("org-mirror",))
        self.assertIn("is_org_mirror_path", stubs.calls)

    def test_multiple_reasons_merge_sorted(self):
        self.make_skill("multi", "multi")
        with self.hermes(unmanaged=[row("multi", provenance="bundled", pinned=True)],
                         cron_names={"multi"}):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts[0].protected_reasons,
                         ("bundled", "cron-referenced", "pinned"))

    def test_cron_referenced_flag(self):
        self.make_skill("cronny", "cronny")
        self.make_skill("free", "free")
        with self.hermes(unmanaged=[row("cronny"), row("free")],
                         cron_names={"cronny"}) as stubs:
            artifacts = self.collect(include_unmanaged=True)
        by_name = {item.name: item for item in artifacts}
        self.assertEqual(by_name["cronny"].protected_reasons, ("cron-referenced",))
        self.assertEqual(by_name["free"].protected_reasons, ())
        self.assertIn("referenced_skill_names", stubs.calls)

    def test_cron_lookup_failure_fails_open_without_crash(self):
        # Mirrors the real referenced_skill_names, which returns an empty set on
        # a corrupt store; the trade-off is that cron protection silently drops.
        self.make_skill("cronny", "cronny")
        with self.hermes(unmanaged=[row("cronny")],
                         cron_error=RuntimeError("corrupt jobs store")):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual([item.name for item in artifacts], ["cronny"])
        self.assertEqual(artifacts[0].protected_reasons, ())

    def test_core_ineligible_flag(self):
        self.make_skill("core", "core")
        self.make_skill("local", "local")
        with self.hermes(unmanaged=[row("core"), row("local")], eligible=False):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual([item.protected_reasons for item in artifacts],
                         [("core-ineligible",), ("core-ineligible",)])
        with self.hermes(unmanaged=[row("core"), row("local")], eligible=True):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual([item.protected_reasons for item in artifacts], [(), ()])

    def test_eligibility_error_flagged_unknown(self):
        self.make_skill("core", "core")
        with self.hermes(unmanaged=[row("core")],
                         eligibility_error=RuntimeError("eligibility backend down")):
            artifacts = self.collect(include_unmanaged=True)
        self.assertEqual(artifacts[0].protected_reasons, ("eligibility-unknown",))
        self.assertTrue(artifacts[0].protected)

    def test_clean_skill_unprotected(self):
        package = self.make_skill("clean", "clean")
        with self.hermes(unmanaged=[row("clean")]):
            artifacts = self.collect(include_unmanaged=True)
        artifact = artifacts[0]
        self.assertEqual(artifact.protected_reasons, ())
        self.assertFalse(artifact.protected)
        self.assertEqual(artifact.path, package)
        self.assertEqual(artifact.digest, expected_digest(package))


class DigestsMatchTests(unittest.TestCase):
    @staticmethod
    def artifact(name: str, digest: str) -> SkillArtifact:
        return SkillArtifact(name=name, path=Path("/skills") / name, description="",
                             text="", digest=digest)

    def test_all_present_and_equal_matches(self):
        inventory_rows = [self.artifact("a", "d1"), self.artifact("b", "d2")]
        self.assertTrue(inventory.digests_match({"a": "d1", "b": "d2"}, inventory_rows))

    def test_extra_inventory_entries_ignored(self):
        inventory_rows = [self.artifact("a", "d1"), self.artifact("b", "d2")]
        self.assertTrue(inventory.digests_match({"a": "d1"}, inventory_rows))

    def test_unverifiable_expectations_never_match(self):
        inventory_rows = [self.artifact("a", "d1")]
        self.assertFalse(inventory.digests_match({}, inventory_rows))
        self.assertFalse(inventory.digests_match(None, inventory_rows))
        self.assertFalse(inventory.digests_match({"missing": "d1"}, inventory_rows))
        self.assertFalse(inventory.digests_match({"a": "other"}, inventory_rows))

    def test_generator_inventory_supported(self):
        self.assertTrue(inventory.digests_match(
            {"a": "d1"}, (item for item in [self.artifact("a", "d1")])))

    def test_none_expected_value_is_not_a_match(self):
        # An unverifiable digest (None) must never satisfy the gate; the current
        # implementation compares None == None for names missing from inventory.
        inventory_rows = [self.artifact("a", "d1")]
        self.assertFalse(inventory.digests_match({"ghost": None}, inventory_rows))
        self.assertFalse(inventory.digests_match({"a": "d1", "ghost": None}, inventory_rows))


if __name__ == "__main__":
    unittest.main()
