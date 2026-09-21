"""Hermes registration surface for the Jev skill-relations curator.

Registers exactly:
  * the ``jev_skill_relations`` tool in the ``skills`` toolset (read-only judgments);
  * a curator-platform-only system prompt section;
  * an ``on_skill_lifecycle`` observer that audits mutations and debounces automatic dry runs;
  * the top-level ``hermes jev-curator`` CLI command and the ``/jev-curator`` slash command,
    parsed and rendered by ``plugin/commands.py`` over the ``plugin/service.py`` facade.

Invariants: default mode is ``observe``; registration mutates nothing; every handler
returns bounded text/JSON and never raises; package imports stay inside handlers, so the
module imports and ``register(ctx)`` runs in the bare ``hermes plugins validate`` probe.
``plugin/plugin.yaml`` declares exactly this surface — validation diffs the manifest
against what ``register()`` actually registers, so keep the two in step.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

TOOL_NAME = "jev_skill_relations"
TOOLSET = "skills"
HOOK_NAME = "on_skill_lifecycle"
GUARD_HOOK_NAME = "pre_tool_call"
COMMAND_NAME = "jev-curator"
SECTION_ID = "jev-curator"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Read-only Jev relation judgments between curator-managed skills. With no arguments it "
        "scans candidate pairs; pass skill to focus on one skill's pairs, or pair=[a, b] to judge "
        "exactly two skills. Returns relations (duplicate, subset, same_class, complementary, "
        "conflict, unrelated, insufficient_evidence) with coverage and preservation evidence. "
        "Judgments are evidence, never authorization: this tool never mutates skills."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Focus the scan on pairs involving this skill."},
            "pair": {
                "type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2,
                "description": "Judge exactly these two skills, e.g. [\"skill-a\", \"skill-b\"].",
            },
            "use_jev": {"type": "boolean", "description": "Ask Jev for judgments (default true)."},
        },
        "required": [],
    },
}

_CURATOR_SECTION = (
    "`jev_skill_relations` (skills toolset) returns read-only relation judgments between "
    "curator-managed skills: duplicate / a_subset_of_b / b_subset_of_a / same_class / "
    "complementary / conflict / unrelated / insufficient_evidence, each with coverage, "
    "preservation, and conflict evidence plus the deterministic lexical baseline.\n\n"
    "Call it before merging, absorbing, or deleting a skill:\n"
    "  - pair: [\"skill-a\", \"skill-b\"] judges exactly two skills;\n"
    "  - skill: \"name\" scans that skill's candidate pairs;\n"
    "  - no arguments scans the whole managed inventory.\n\n"
    "Treat judgments as evidence, not authorization. `insufficient_evidence` or low coverage "
    "means read both skills yourself before acting. The tool only reports: it never writes, "
    "and nothing it returns approves a mutation."
)

_CTX: Any = None
_CTX_CONFIG_OK: bool | None = None
_DEBOUNCER: Any = None


# -- handlers --------------------------------------------------------------------


def _tool_handler(args: Mapping[str, Any] | None = None, **kwargs: Any) -> str:
    """Tool entry point: always a JSON string, never an exception."""
    try:
        return _json(_tool_payload(dict(args or {})))
    except Exception as exc:
        logger.warning("jev-curator tool failed: %s", exc)
        return _json({"ok": False, "error": _describe(exc)})


def _tool_payload(args: Mapping[str, Any]) -> dict[str, Any]:
    engine = _service().engine
    pair = args.get("pair")
    if pair is not None:
        first, second = _pair_names(pair)
        return engine.review(first, second)
    name = str(args.get("skill") or "").strip()
    if name:
        return engine.review(name)
    return engine.scan(use_jev=bool(args.get("use_jev", True)))


def _on_skill_lifecycle(**event: Any) -> None:
    """Audit lifecycle facts and debounce a profile-scoped dry run; never raise."""
    try:
        action = str(event.get("action") or "").strip()
        if not action or not str(event.get("skill_name") or "").strip():
            return
        service = _service()
        service.lifecycle(**event)
        if _DEBOUNCER is not None:
            _DEBOUNCER.notify(action, service=service)
    except Exception:
        logger.debug("jev-curator lifecycle observer failed", exc_info=True)


def _curator_prompt_section(session_info: Mapping[str, Any] | None = None) -> str:
    """Render only for the curator fork (``platform="curator"``); empty everywhere else."""
    try:
        platform = str((session_info or {}).get("platform") or "").strip().lower()
    except Exception:
        return ""
    return _CURATOR_SECTION if platform == "curator" else ""


# -- shared wiring ---------------------------------------------------------------


def _service() -> Any:
    """Profile-scoped service facade; package imports stay lazy."""
    from .service import build_service

    global _CTX_CONFIG_OK
    service = build_service(_CTX)
    _CTX_CONFIG_OK = True
    return service


def _wire_commands(ctx: Any) -> None:
    """Hand both command surfaces to plugin/commands.py; it owns parsing and rendering."""
    from . import commands

    commands.set_service_factory(_service)
    ctx.register_cli_command(
        name=COMMAND_NAME,
        help="Inspect Jev skill-relation judgments (read-only)",
        setup_fn=commands.setup_cli_parser,
        handler_fn=commands.cli_handler,
        description="Read-only Jev relation judgments over curator-managed skills.",
    )
    ctx.register_command(
        COMMAND_NAME, commands.make_slash_handler(None),
        description="Jev skill-relation judgments over curator-managed skills.",
        args_hint="[status|scan|review|graph|plan|run|doctor]",
    )


def _pair_names(pair: Any) -> tuple[str, str]:
    names = [str(item).strip() for item in pair] if isinstance(pair, (list, tuple)) else []
    if len(names) != 2 or not all(names):
        raise ValueError("pair must be exactly two skill names")
    return names[0], names[1]


def _json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"ok": False, "error": "payload is not JSON-serializable"})


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


# -- entry point -----------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register Jev relation evidence and the opt-in background mutation guard."""
    global _CTX, _CTX_CONFIG_OK, _DEBOUNCER
    _CTX = ctx
    _CTX_CONFIG_OK = None
    if _DEBOUNCER is not None:
        _DEBOUNCER.close()
    from .debounce import LifecycleDebouncer
    _DEBOUNCER = LifecycleDebouncer(_service)
    ctx.on_unload(_DEBOUNCER.close)
    ctx.register_tool(
        name=TOOL_NAME, toolset=TOOLSET, schema=TOOL_SCHEMA, handler=_tool_handler,
        description="Read-only Jev relation judgments between curator-managed skills",
        emoji="⚖️",
    )
    ctx.register_system_prompt_section(SECTION_ID, _curator_prompt_section, position="after_memory")
    ctx.register_hook(HOOK_NAME, _on_skill_lifecycle)
    from .guard import make_pre_tool_call_hook
    ctx.register_hook(GUARD_HOOK_NAME, make_pre_tool_call_hook(_service))
    _wire_commands(ctx)
