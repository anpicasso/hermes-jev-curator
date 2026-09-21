"""Synthetic registration test: a stub ctx, no host, no network, no filesystem writes.

Mirrors what ``hermes plugins validate``'s capability probe does (import the plugin,
call ``register(ctx)`` against a recording stub, diff against the manifest), plus the
never-raise contract for every handler and the curator-only prompt section.
"""

from __future__ import annotations

import argparse
import contextlib
import asyncio
import dataclasses
import io
import json
import unittest
from pathlib import Path
from unittest import mock

import plugin
from plugin import commands
from plugin.models import Settings

try:
    import yaml
except ImportError:  # pragma: no cover - the repo venv always has PyYAML
    yaml = None

MANIFEST = Path(__file__).resolve().parents[1] / "plugin.yaml"
CONFIG_TYPES = {"str", "string", "int", "integer", "float", "number", "bool", "boolean",
                "list", "array", "dict", "mapping", "map"}


class StubContext:
    """Recording ctx mirroring the probe: real registration methods only, the rest raises."""

    def __init__(self, config=None):
        self.calls = []
        self.unloads = []
        self.config = dict(config or {})

    def register_tool(self, **kwargs):
        self.calls.append(("tool", kwargs))

    def register_system_prompt_section(self, section_id, content, **kwargs):
        self.calls.append(("section", {"id": section_id, "content": content, **kwargs}))

    def register_hook(self, name, callback):
        self.calls.append(("hook", {"name": name, "callback": callback}))

    def register_cli_command(self, **kwargs):
        self.calls.append(("cli", kwargs))

    def register_command(self, name, handler, **kwargs):
        self.calls.append(("slash", {"name": name, "handler": handler, **kwargs}))

    def on_unload(self, callback):
        self.unloads.append(callback)

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def __getattr__(self, name):  # an accidental extra registration surface fails loudly
        raise AttributeError(name)

    def of(self, kind):
        return [payload for call, payload in self.calls if call == kind]


class FakeEngine:
    def status(self):
        return {"ok": True, "mode": "observe", "managed_skills": 2}

    def scan(self, names=None, *, use_jev=True):
        return {"ok": True, "mode": "observe", "candidates": [], "judgments": [], "errors": []}

    def review(self, first, second=None):
        return {"ok": True, "judgment": {"a": first, "b": second or first,
                                         "relation": "unrelated", "confidence": 0.5}}


class FakeService:
    settings = Settings()
    engine = FakeEngine()

    def __init__(self):
        self.events = []

    def status(self):
        return self.engine.status()

    def scan(self):
        return self.engine.scan()

    def review(self, name):
        return self.engine.review(name)

    def graph(self):
        return {"ok": True, "nodes": []}

    def plan(self):
        return {"ok": True, "plans": []}

    def run(self, *, apply=False):
        return {"ok": True, "applied": []}

    def doctor(self):
        return {"ok": True, "checks": []}

    def lifecycle(self, **event):
        self.events.append(event)


class BoomService:
    @property
    def engine(self):
        raise RuntimeError("engine exploded")

    def lifecycle(self, **event):
        raise RuntimeError("audit exploded")


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.ctx = StubContext()
        plugin.register(self.ctx)

    def test_registers_exactly_the_declared_surface(self):
        self.assertEqual({call for call, _ in self.ctx.calls},
                         {"tool", "section", "hook", "cli", "slash"})
        tool = self.ctx.of("tool")[0]
        self.assertEqual((tool["name"], tool["toolset"]), ("jev_skill_relations", "skills"))
        self.assertIs(tool["handler"], plugin._tool_handler)
        self.assertEqual({row["name"] for row in self.ctx.of("hook")},
                         {"on_skill_lifecycle", "pre_tool_call"})
        cli = self.ctx.of("cli")[0]
        self.assertEqual(cli["name"], "jev-curator")
        self.assertIs(cli["setup_fn"], commands.setup_cli_parser)
        self.assertIs(cli["handler_fn"], commands.cli_handler)
        self.assertEqual(self.ctx.of("slash")[0]["name"], "jev-curator")
        section = self.ctx.of("section")[0]
        self.assertEqual((section["id"], section["position"]), ("jev-curator", "after_memory"))
        self.assertEqual(len(self.ctx.unloads), 1)

    @unittest.skipUnless(yaml, "PyYAML unavailable")
    def test_manifest_matches_registration(self):
        manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual((manifest["name"], manifest["version"], manifest["kind"]),
                         ("jev-curator", "0.1.0", "standalone"))
        self.assertEqual(set(manifest["provides_tools"]), {plugin.TOOL_NAME})
        self.assertEqual(set(manifest["provides_hooks"]),
                         {plugin.HOOK_NAME, plugin.GUARD_HOOK_NAME})
        self.assertEqual(manifest["requires_hermes"], ">=0.21.2")
        self.assertFalse(manifest.get("capabilities"))  # no privileged capability is used
        settings_keys = {field.name for field in dataclasses.fields(Settings)}
        settings_keys.remove("model")
        settings_keys.add("jev_model")
        self.assertEqual(set(manifest["config_schema"]), settings_keys)
        for key, spec in manifest["config_schema"].items():
            self.assertIn(str(spec["type"]).lower(), CONFIG_TYPES, key)

    def test_default_mode_is_observe(self):
        from plugin.service import build_service

        self.assertEqual(build_service(StubContext()).settings.mode, "observe")
        self.assertEqual(build_service(StubContext({"mode": "apply"})).settings.mode, "apply")
        self.assertEqual(build_service(StubContext({"mode": "nonsense"})).settings.mode, "observe")

    def test_service_uses_non_reserved_jev_model_key(self):
        with mock.patch.object(plugin, "_CTX", StubContext({"jev_model": "jev-pinned"})):
            plugin._CTX_CONFIG_OK = None
            service = plugin._service()
            self.assertTrue(plugin._CTX_CONFIG_OK)
            self.assertEqual(service.settings.model, "jev-pinned")
        plugin._CTX_CONFIG_OK = None

    def test_section_is_curator_platform_only(self):
        content = self.ctx.of("section")[0]["content"]
        self.assertEqual(content({"platform": "curator"}), plugin._CURATOR_SECTION)
        self.assertEqual(content({"platform": " Curator "}), plugin._CURATOR_SECTION)
        for info in ({}, {"platform": "cli"}, {"platform": "gateway"}, None):
            self.assertEqual(content(info), "")
        self.assertLess(len(plugin._CURATOR_SECTION), 4000)

    def test_tool_handler_never_raises(self):
        with mock.patch.object(plugin, "_service", lambda: FakeService()):
            self.assertTrue(json.loads(plugin._tool_handler({}))["ok"])
            pair = json.loads(plugin._tool_handler({"pair": ["a", "b"]}, task_id="t"))
            self.assertEqual((pair["judgment"]["a"], pair["judgment"]["b"]), ("a", "b"))
            self.assertTrue(json.loads(plugin._tool_handler({"skill": "a"}))["ok"])
            for bad in ({"pair": ["only-one"]}, {"pair": "a"}, {"pair": [1, 2, 3]}):
                self.assertFalse(json.loads(plugin._tool_handler(bad))["ok"], bad)
        with mock.patch.object(plugin, "_service", lambda: BoomService()):
            payload = json.loads(plugin._tool_handler({}))
            self.assertFalse(payload["ok"])
            self.assertIn("engine exploded", payload["error"])
        with mock.patch.object(plugin, "_service", mock.Mock(side_effect=ImportError("no service"))):
            self.assertFalse(json.loads(plugin._tool_handler({}))["ok"])

    def test_observer_records_and_never_raises(self):
        service = FakeService()
        notifier = mock.Mock()
        with mock.patch.object(plugin, "_service", lambda: service), \
             mock.patch.object(plugin, "_DEBOUNCER", notifier):
            plugin._on_skill_lifecycle(action="patched", skill_name="x", task_id="t", session_id="s")
            for event in ({}, {"action": "", "skill_name": "x"}, {"action": "patched"}, None):
                if event is not None:
                    plugin._on_skill_lifecycle(**event)
        self.assertEqual([row["skill_name"] for row in service.events], ["x"])
        self.assertEqual(service.events[0]["action"], "patched")
        notifier.notify.assert_called_once_with("patched", service=service)
        with mock.patch.object(plugin, "_service", lambda: BoomService()):
            plugin._on_skill_lifecycle(action="patched", skill_name="x")  # must not raise

    def test_command_surfaces_answer_without_raising(self):
        self.assertIsNotNone(commands._SERVICE_FACTORY)  # register() wired the service seam
        service = FakeService()
        commands.set_service_factory(lambda: service)
        slash = self.ctx.of("slash")[0]["handler"]
        self.assertIn("jev-curator commands:", asyncio.run(slash("")))
        self.assertIn("ok", asyncio.run(slash("status")))
        self.assertIn("unrelated", asyncio.run(slash("review alpha")))
        self.assertIn("refused", asyncio.run(slash("run --apply")))
        for text in ("bogus", "review", "scan --nope", "review 'unclosed"):
            self.assertIsInstance(asyncio.run(slash(text)), str)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = commands.cli_handler(argparse.Namespace(command="status", json=True), service=service)
        self.assertEqual(code, 0)
        self.assertIn("ok", out.getvalue())


if __name__ == "__main__":
    unittest.main()
