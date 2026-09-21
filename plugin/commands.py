"""CLI and slash-command surface for the Jev curator plugin.

Parsing only. Every handler returns bounded text/JSON and never raises.
`--apply` needs an explicit flag *and* ``settings.mode == "apply"``; slash text
is always dry-run, so a stray chat message can never mutate skills.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from collections.abc import Mapping
from typing import Any, Callable, NoReturn, Protocol

from .models import Settings


_MAX_OUTPUT_CHARS = 12_000
_MAX_ROWS = 20
_MAX_KEYS = 30
_MAX_CELL_CHARS = 300
_MAX_ERROR_CHARS = 400

_COMMANDS: dict[str, str] = {
    "status": "curator mode, inventory size, last-run summary",
    "scan": "collect inventory and deterministic candidate pairs",
    "review": "review NAME: one skill plus its candidate neighbors",
    "graph": "candidate graph summary",
    "plan": "proposed merge plans and blockers",
    "run": "run the pipeline dry; --apply executes when mode=apply",
    "doctor": "configuration, provider, and safety-gate checks",
}

_SERVICE_FACTORY: Callable[[], Any] | None = None


class CuratorService(Protocol):
    """What the command surface needs from a curator service.

    Methods return JSON-able mappings; output is bounded here, so they may
    return anything. ``run(apply=False)`` must never mutate anything.
    """

    settings: Settings

    def status(self) -> Mapping[str, Any]: ...
    def scan(self) -> Mapping[str, Any]: ...
    def review(self, name: str) -> Mapping[str, Any]: ...
    def graph(self) -> Mapping[str, Any]: ...
    def plan(self) -> Mapping[str, Any]: ...
    def run(self, *, apply: bool = False) -> Mapping[str, Any]: ...
    def doctor(self) -> Mapping[str, Any]: ...


def set_service_factory(factory: Callable[[], Any] | None) -> None:
    """Wire how `cli_handler` obtains a service; the plugin entry point calls this."""
    global _SERVICE_FACTORY
    _SERVICE_FACTORY = factory


def setup_cli_parser(parser: Any) -> Any:
    """Add curator subcommands to `parser` (or to a subparsers action directly)."""
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND") if hasattr(parser, "add_subparsers") else parser
    for name, help_text in _COMMANDS.items():
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--json", action="store_true", help="bounded JSON output")
        if name == "review":
            child.add_argument("name", metavar="NAME", help="skill name to review")
        if name == "run":
            child.add_argument("--apply", action="store_true", help="execute mutations; needs mode=apply")
    return parser


def cli_handler(args: Any, service: Any | None = None) -> int:
    """Print bounded output for `args`; return an exit code. Never raises."""
    text, code = _execute(args, service)
    print(text)
    return code


def make_slash_handler(service: Any) -> Callable[[str | None], Any]:
    """Build an async chat handler; slow work never blocks the gateway event loop."""
    parser = _SlashParser(prog="jev-curator", add_help=False)
    setup_cli_parser(parser)

    async def handler(text: str | None = None) -> str:
        try:
            tokens = shlex.split(text or "")
        except ValueError as exc:
            return _clip(f"jev-curator: could not parse command text: {exc}", _MAX_ERROR_CHARS)
        if not tokens or tokens[0].lower() in {"help", "?", "-h", "--help"}:
            return _help_text()
        if any(token in {"-h", "--help"} for token in tokens[1:]):
            return _help_text()
        try:
            args = parser.parse_args(tokens)
        except (_UsageError, SystemExit) as exc:
            message = getattr(exc, "message", None) or exc
            return _clip(f"jev-curator: {message}", _MAX_ERROR_CHARS)
        if getattr(args, "command", "") == "run" and getattr(args, "apply", False):
            return ("jev-curator: --apply is refused from chat. Run "
                    "`hermes jev-curator run --apply` in a terminal with mode=apply.")
        return (await asyncio.to_thread(_execute, args, service))[0]

    return handler


def _execute(args: Any, service: Any = None) -> tuple[str, int]:
    command = str(getattr(args, "command", "") or getattr(args, "subcommand", "") or "").strip().lower()
    if command not in _COMMANDS:
        return _clip(f"jev-curator: unknown command {command!r}; expected {', '.join(_COMMANDS)}", _MAX_ERROR_CHARS), 2
    if service is None:
        try:
            service = _resolve_service()
        except Exception as exc:
            return _safe_error("service unavailable", exc), 1
    try:
        if command == "run" and bool(getattr(args, "apply", False)):
            mode = _mode_of(service)
            if mode != "apply":
                return (f"jev-curator run --apply refused: mode is {mode or 'unknown'}, not 'apply'. "
                        "Set mode: apply and invoke `hermes jev-curator run --apply` explicitly."), 1
        if command == "review":
            name = str(getattr(args, "name", "") or "").strip()
            if not name:
                return "jev-curator review: NAME is required", 2
            data = service.review(name)
        elif command == "run":
            data = service.run(apply=bool(getattr(args, "apply", False)))
        else:
            method = getattr(service, command, None)
            if not callable(method):
                return f"jev-curator: service has no {command}()", 1
            data = method()
        code = 1 if isinstance(data, Mapping) and data.get("ok") is False else 0
        return _format(data, bool(getattr(args, "json", False))), code
    except Exception as exc:
        return _safe_error(command, exc), 1


def _resolve_service() -> Any:
    if _SERVICE_FACTORY is not None:
        return _SERVICE_FACTORY()
    from .service import build_service  # ponytail: one convention; use set_service_factory if the impl differs
    return build_service()


def _mode_of(service: Any) -> str:
    settings = getattr(service, "settings", None)
    mode = getattr(settings, "mode", None) or getattr(service, "mode", "")
    return str(mode or "").strip().lower()


class _UsageError(ValueError):
    pass


class _SlashParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # no SystemExit, no stderr noise in chat
        raise _UsageError(message)


def _help_text() -> str:
    lines = ["jev-curator commands:"]
    lines += [f"  {name:7s} {_COMMANDS[name]}" for name in _COMMANDS]
    lines.append("  --json   bounded JSON output (any command)")
    return "\n".join(lines)


def _format(data: Any, as_json: bool) -> str:
    if as_json:
        rendered = _json(data)
        if len(rendered) > _MAX_OUTPUT_CHARS:
            return _json({"ok": False, "truncated": True,
                          "error": f"output exceeds {_MAX_OUTPUT_CHARS} characters; narrow the query"})
        return rendered
    return _render_text(data)


def _safe_error(scope: str, exc: Exception) -> str:
    try:
        from .state import redact_text
        detail = redact_text(str(exc))
    except Exception:
        detail = "details unavailable"
    return _clip(f"jev-curator {scope}: {type(exc).__name__}: {detail}", _MAX_ERROR_CHARS)


def _render_text(data: Any) -> str:
    if not isinstance(data, Mapping):
        return _clip(_json(data), _MAX_OUTPUT_CHARS)
    items = list(data.items())
    lines: list[str] = []
    for key, value in items[:_MAX_KEYS]:
        lines.extend(_render_field(str(key), value))
    if len(items) > _MAX_KEYS:
        lines.append(f"… +{len(items) - _MAX_KEYS} more keys")
    return _clip("\n".join(lines) or "(empty)", _MAX_OUTPUT_CHARS)


def _render_field(key: str, value: Any) -> list[str]:
    if isinstance(value, Mapping):
        inner = list(value.items())
        rows = [f"  {name}: {_cell(item)}" for name, item in inner[:_MAX_KEYS]]
        if len(inner) > _MAX_KEYS:
            rows.append(f"  … +{len(inner) - _MAX_KEYS} more")
        return [f"{key}:"] + rows
    if isinstance(value, (list, tuple)):
        rows = [f"  - {_cell(item)}" for item in list(value)[:_MAX_ROWS]]
        if len(value) > _MAX_ROWS:
            rows.append(f"  … +{len(value) - _MAX_ROWS} more")
        return [f"{key} ({len(value)}):"] + rows
    return [f"{key}: {_cell(value)}"]


def _cell(value: Any) -> str:
    if isinstance(value, str):
        return _clip(value, _MAX_CELL_CHARS)
    if value is None or isinstance(value, (int, float, bool)):
        return str(value)
    return _clip(_compact(value), _MAX_CELL_CHARS)


def _compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def _clip(text: str, limit: int) -> str:
    marker = "\n… [explicitly truncated] …\n"
    if len(text) <= limit:
        return text
    room = max(1, limit - len(marker))
    head = room * 2 // 3
    tail = room - head
    return text[:head] + marker + (text[-tail:] if tail else "")
