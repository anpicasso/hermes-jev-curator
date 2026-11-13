"""The host symbols this plugin's inventory needs must exist in the RUNNING Hermes.

`test_inventory.py` swaps in fakes so the parser can be tested in isolation, and a fake
kept a symbol the host had already deleted (`agent.skill_utils.is_org_mirror_path`,
gone in 0.23) — every real inventory call raised ImportError while the suite stayed green.
This module imports the real host instead, so a moved/removed symbol fails here first.

Run with the host on sys.path (``hermes plugins doctor`` / the venv that runs Hermes).
No network, no profile writes: ``collect_inventory`` is read-only.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN = Path(__file__).resolve().parent.parent
if str(PLUGIN.parent) not in sys.path:
    sys.path.insert(0, str(PLUGIN.parent))

try:
    import agent.skill_utils  # noqa: F401
    import hermes_constants  # noqa: F401
    import tools.skill_usage  # noqa: F401
except ImportError as exc:  # pragma: no cover - bare interpreter
    raise unittest.SkipTest(f"Hermes core not importable: {exc}")

from plugin import inventory  # noqa: E402


class HostContractTests(unittest.TestCase):
    def test_inventory_host_symbols_exist(self):
        expected = {
            "agent.skill_utils": ("ESSENTIAL_SKILLS", "get_disabled_skill_names",
                                  "iter_skill_index_files", "parse_frontmatter"),
            "tools.skill_usage": ("curated_report", "is_curation_eligible",
                                  "provenance", "usage_report"),
            "tools.skill_provenance": ("is_background_review",),
            "cron.jobs": ("referenced_skill_names",),
        }
        for module_name, names in expected.items():
            module = importlib.import_module(module_name)
            missing = [name for name in names if not hasattr(module, name)]
            self.assertEqual(missing, [], f"{module_name} no longer provides {missing}")

    def test_real_collect_inventory_runs(self):
        """The exact call the tool/guard make, against the live host modules."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "skills"
            (root / "sample").mkdir(parents=True)
            (root / "sample" / "SKILL.md").write_text(
                "---\nname: sample\ndescription: synthetic\n---\nBody.\n", encoding="utf-8")
            artifacts = inventory.collect_inventory(skills_root=root, names={"sample"})
        self.assertIsInstance(artifacts, list)

    def test_org_mirror_layout_is_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "skills"
            package = root / "_org" / "acme" / "shared"
            package.mkdir(parents=True)
            self.assertTrue(inventory._is_org_mirror(package, root))
            self.assertFalse(inventory._is_org_mirror(root / "plain", root))


if __name__ == "__main__":
    unittest.main()
