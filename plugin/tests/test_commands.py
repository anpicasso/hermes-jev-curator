from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import threading
import unittest

from plugin.commands import (
    _MAX_OUTPUT_CHARS,
    cli_handler,
    make_slash_handler,
    set_service_factory,
    setup_cli_parser,
)
from plugin.models import Settings


class FakeService:
    """Records calls; optionally raises. Mirrors the CuratorService protocol."""

    def __init__(self, *, mode: str = "observe", payload=None, error: Exception | None = None):
        self.settings = Settings(mode=mode)
        self.calls: list[tuple] = []
        self.payload = {"mode": mode, "count": 3} if payload is None else payload
        self.error = error

    def _call(self, *args):
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.payload

    def status(self):
        return self._call("status")

    def scan(self):
        return self._call("scan")

    def review(self, name):
        return self._call("review", name)

    def graph(self):
        return self._call("graph")

    def plan(self):
        return self._call("plan")

    def run(self, *, apply=False):
        return self._call("run", apply)

    def doctor(self):
        return self._call("doctor")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-curator")
    setup_cli_parser(parser)
    return parser


def run_cli(service, argv) -> tuple[int, str]:
    args = make_parser().parse_args(argv)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli_handler(args, service)
    return code, buffer.getvalue()


def run_slash(handler, text):
    return asyncio.run(handler(text))


class CommandTests(unittest.TestCase):
    def test_parser_accepts_every_command(self):
        parser = make_parser()
        self.assertEqual(parser.parse_args(["status"]).command, "status")
        self.assertEqual(parser.parse_args(["scan"]).command, "scan")
        self.assertEqual(parser.parse_args(["review", "git"]).name, "git")
        self.assertEqual(parser.parse_args(["graph"]).command, "graph")
        self.assertEqual(parser.parse_args(["plan"]).command, "plan")
        self.assertEqual(parser.parse_args(["doctor"]).command, "doctor")
        self.assertFalse(parser.parse_args(["run"]).apply)
        self.assertTrue(parser.parse_args(["run", "--apply"]).apply)
        self.assertTrue(parser.parse_args(["status", "--json"]).json)

    def test_cli_dispatches_every_command(self):
        service = FakeService()
        for argv in (["status"], ["scan"], ["review", "git"], ["graph"], ["plan"], ["run"], ["doctor"]):
            code, out = run_cli(service, argv)
            self.assertEqual(code, 0, out)
            self.assertTrue(out.strip())
        self.assertEqual([call[0] for call in service.calls],
                         ["status", "scan", "review", "graph", "plan", "run", "doctor"])
        self.assertIn(("review", "git"), service.calls)

    def test_apply_needs_mode_and_flag(self):
        observe = FakeService(mode="observe")
        code, out = run_cli(observe, ["run", "--apply"])
        self.assertEqual(code, 1)
        self.assertIn("refused", out)
        self.assertNotIn(("run", True), observe.calls)
        run_cli(observe, ["run"])
        self.assertIn(("run", False), observe.calls)

        armed = FakeService(mode="apply")
        code, _ = run_cli(armed, ["run", "--apply"])
        self.assertEqual(code, 0)
        self.assertIn(("run", True), armed.calls)

        armed_dry = FakeService(mode="apply")
        run_cli(armed_dry, ["run"])
        self.assertEqual([call for call in armed_dry.calls if call[0] == "run"], [("run", False)])

    def test_explicit_failed_result_returns_nonzero(self):
        code, out = run_cli(FakeService(payload={"ok": False, "error": "refused"}), ["status"])
        self.assertEqual(code, 1)
        self.assertIn("refused", out)

    def test_slash_refuses_apply_and_never_raises(self):
        service = FakeService(mode="apply")
        handler = make_slash_handler(service)
        for text in ("run --apply", "run  --apply", "apply", "run --apply now", "review --apply"):
            out = run_slash(handler, text)
            self.assertIsInstance(out, str)
            self.assertTrue(out.strip())
        self.assertNotIn(("run", True), service.calls)
        self.assertIn("refused", run_slash(handler, "run --apply").lower())

        for text in (None, "", "   ", "???", "not-a-command", "review 'unclosed", "run --nope", "run extra"):
            out = run_slash(handler, text)
            self.assertTrue(out.strip(), text)
        self.assertNotIn(("run", True), service.calls)

    def test_slash_runs_dry_and_lists_help(self):
        service = FakeService(mode="apply")
        handler = make_slash_handler(service)
        run_slash(handler, "run")
        self.assertEqual([call for call in service.calls if call[0] == "run"], [("run", False)])
        self.assertIn("status", run_slash(handler, ""))
        self.assertIn("status", run_slash(handler, "help"))
        self.assertIn("status", run_slash(handler, "status --help"))
        run_slash(handler, "review my-skill")
        self.assertIn(("review", "my-skill"), service.calls)

    def test_slash_dispatches_service_work_off_the_event_loop_thread(self):
        service = FakeService()
        worker_threads = []

        def scan():
            worker_threads.append(threading.get_ident())
            return {"ok": True}

        service.scan = scan
        caller_thread = threading.get_ident()
        self.assertIn("ok", run_slash(make_slash_handler(service), "scan"))
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], caller_thread)

    def test_service_errors_become_bounded_text(self):
        service = FakeService(error=RuntimeError("boom " + "z" * 5_000))
        code, out = run_cli(service, ["status"])
        self.assertEqual(code, 1)
        self.assertIn("RuntimeError", out)
        self.assertLessEqual(len(out), 500)
        self.assertTrue(run_slash(make_slash_handler(service), "status").strip())

    def test_output_is_bounded_and_json_parses(self):
        huge = {
            "mode": "observe",
            "blob": "x" * 500_000,
            "rows": [{"n": index, "text": "y" * 1_000} for index in range(500)],
        }
        service = FakeService(payload=huge)
        for argv in (["status"], ["status", "--json"]):
            code, out = run_cli(service, argv)
            self.assertEqual(code, 0)
            self.assertLessEqual(len(out), _MAX_OUTPUT_CHARS + 1)
            self.assertIn("truncated", out)
            if "--json" in argv:
                self.assertTrue(json.loads(out)["truncated"])
        code, small = run_cli(FakeService(), ["status", "--json"])
        self.assertEqual(json.loads(small), {"mode": "observe", "count": 3})

    def test_subcommand_help_is_returned_without_argparse_stdout(self):
        handler = make_slash_handler(FakeService())
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            out = run_slash(handler, "status --help")
        self.assertEqual(buffer.getvalue(), "")
        self.assertIn("jev-curator commands", out)

    def test_formatting_failures_are_contained(self):
        class BadString:
            def __str__(self):
                raise RuntimeError("cannot stringify")

            __repr__ = __str__

        code, out = run_cli(FakeService(payload={"bad": BadString()}), ["status", "--json"])
        self.assertEqual(code, 1)
        self.assertIn("RuntimeError", out)

    def test_bad_args_never_raise(self):
        for args in (argparse.Namespace(command="bogus"), argparse.Namespace(command="review"),
                     argparse.Namespace(command="review", name="   "), None):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = cli_handler(args, FakeService())
            self.assertNotEqual(code, 0)
            self.assertTrue(buffer.getvalue().strip())

    def test_service_factory_seam(self):
        set_service_factory(lambda: FakeService())
        try:
            args = make_parser().parse_args(["status"])
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = cli_handler(args)
            self.assertEqual(code, 0)
            self.assertIn("observe", buffer.getvalue())
        finally:
            set_service_factory(None)


if __name__ == "__main__":
    unittest.main()
