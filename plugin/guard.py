"""Deterministic local mutation guard for background curator skill writes.

The parent registers :func:`make_pre_tool_call_hook` as a ``pre_tool_call`` hook; this module
owns the decision. In ``guard``/``apply`` mode a *destructive* ``skill_manage`` call made by
the background review fork is refused unless a hash-bound authorized merge plan installed in
plugin state (``<root>/guard_plans.json``) currently covers the exact package content being
mutated. Everything is local -- plugin state files plus the read-only skill inventory, no
network, no writes outside ``<root>``.

Decision table:

    mode off / observe          -> None  (inert; observe is the default and never interferes)
    foreground (not review fork)-> None  (user-directed calls are never blocked)
    tool != skill_manage        -> None
    create / new-file write_file-> None  (fail open; core's own guards still apply)
    delete, background, guard|apply:
        authorized only when ``absorbed_into`` matches a validated direct-edge plan and the
        current package hashes still match. Destructive content writes are always refused:
        relation plans prove containment, not that proposed replacement bytes preserve rules.

Plans are hash-bound: a plan only authorizes the content it was built from, so any change to a
covered package turns the authorization stale until a fresh plan is installed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

from .graph import pair_key
from .inventory import artifact_host_names, collect_inventory
from .state import audit, read_json, state_root, write_json

_ACTIVE_MODES = frozenset({"guard", "apply"})
_PLANS_FILE = "guard_plans.json"
_PLANS_VERSION = 1
_MAX_PLANS = 64
_MAX_ID_CHARS = 128
_MAX_MESSAGE_CHARS = 400
_MAX_NAME_CHARS = 80
_MAX_DETAIL_CHARS = 200
_DESTRUCTIVE_ACTIONS = frozenset({"delete", "patch", "edit", "remove_file"})


class _Op(NamedTuple):
    """One normalized skill_manage operation (flat call or one batch entry)."""

    action: str
    name: str
    file_path: str
    absorbed_into: str


# --- public API ----------------------------------------------------------------------

def pre_tool_call_guard(settings_or_factory: Any, tool_name: str = "", args: Any = None,
                        **kwargs: Any) -> dict[str, str] | None:
    """Decide one ``pre_tool_call`` event: a block directive, or None to let the call through.

    ``settings_or_factory`` is a ``Settings``, a mapping with a ``mode``, or a zero-arg callable
    returning either (a service with ``.settings`` is accepted too). Unreadable settings leave
    the guard inert. Extra hook kwargs are ignored; this callback never raises.
    """
    try:
        if str(tool_name or "").strip() != "skill_manage":
            return None
        if not _background_review():
            return None
        if _active_mode(settings_or_factory) is None:
            return None
        return _gate(_normalize_ops(args))
    except Exception:
        return None


def make_pre_tool_call_hook(settings_or_factory: Any):
    """Bind the guard to a hook callable: ``ctx.register_hook("pre_tool_call", ...)``."""

    def _hook(**kwargs: Any) -> dict[str, str] | None:
        extra = {key: value for key, value in kwargs.items() if key not in ("tool_name", "args")}
        return pre_tool_call_guard(
            settings_or_factory, kwargs.get("tool_name", ""), kwargs.get("args"), **extra)

    return _hook


def plans_path() -> Path:
    """Location of the hash-bound authorization store inside plugin state."""
    return state_root() / _PLANS_FILE


def install_authorized_plan(plan: Any) -> str | None:
    """Persist one validated merge plan (``MergePlan`` or mapping) as guard authorization.

    Returns the ``plan_id`` on success, ``None`` when the plan is not installable (blocked,
    noop, missing digests, or no direct edge for an absorbed member) or the store is unwritable.
    Upserts by ``plan_id``; the store keeps at most ``_MAX_PLANS`` newest entries. Never raises.
    """
    try:
        entry = _entry_from(plan)
        if entry is None:
            return None
        raw = read_json(plans_path(), {})
        existing = raw.get("plans") if isinstance(raw, Mapping) else {}
        if not isinstance(existing, Mapping):
            existing = {}
        kept = {item["plan_id"]: item for item in
                (_valid_entry(value) for value in existing.values()) if item}
        kept.pop(entry["plan_id"], None)
        kept[entry["plan_id"]] = entry
        if len(kept) > _MAX_PLANS:
            newest = sorted(kept.values(), key=lambda item: (item["installed_at"], item["plan_id"]))
            kept = {item["plan_id"]: item for item in newest[-_MAX_PLANS:]}
        write_json(plans_path(), {"version": _PLANS_VERSION, "plans": kept})
        audit("guard_plan_install", plan_id=entry["plan_id"], canonical=entry["canonical"],
              absorbed=len(entry["absorbed"]))
        return entry["plan_id"]
    except Exception:
        return None


def replace_authorized_plans(plans: Any) -> list[str] | None:
    """Atomically replace guard authority with exactly the current validated plans."""
    try:
        entries = [entry for entry in (_entry_from(plan) for plan in plans) if entry is not None]
        entries = sorted(entries, key=lambda item: (item["installed_at"], item["plan_id"]))[-_MAX_PLANS:]
        kept = {entry["plan_id"]: entry for entry in entries}
        write_json(plans_path(), {"version": _PLANS_VERSION, "plans": kept})
        ids = sorted(kept)
        audit("guard_plans_replace", plans=len(ids))
        return ids
    except Exception:
        return None


def load_authorized_plans() -> list[dict[str, Any]]:
    """Validated, hash-bound plans currently installed; ``[]`` when missing or unreadable."""
    return _load_plan_entries()[0]


# --- decision ------------------------------------------------------------------------

def _gate(ops: list[_Op]) -> dict[str, str] | None:
    """Gate every destructive op in one call; a batch is atomic, so one refusal blocks it."""
    candidates = [op for op in ops if op.action in _DESTRUCTIVE_ACTIONS or op.action == "write_file"]
    if not candidates:
        return None  # creates and unknown actions stay core's business (fail open)
    try:
        plans, problem = _load_plan_entries()
        wanted = {op.name for op in candidates if op.name}
        wanted.update(op.absorbed_into for op in candidates if op.absorbed_into)
        inventory = _current_artifacts(wanted)
        if inventory is None:
            return _block(candidates[0], "the skill inventory is unreadable")
        for op in candidates:
            detail = _op_refusal(op, plans, problem, inventory)
            if detail:
                return _block(op, detail)
        return None
    except Exception:
        return _block(_Op("", "", "", ""), "the guard failed while checking authorization")


def _op_refusal(op: _Op, plans: list[dict[str, Any]], problem: str,
                inventory: Mapping[str, Any]) -> str | None:
    """Refusal detail for one destructive op, or None when it is authorized."""
    name = op.name
    if not name:
        return "the operation carries no skill name"
    artifact = inventory.get(name)
    if op.action == "write_file":
        if artifact is None:
            return None  # core refuses a write to a missing skill; nothing to overwrite
        if not _overwrites(artifact, op.file_path):
            return None  # new file: fail open
    if artifact is None:
        return f"skill '{name}' is not in the managed inventory"
    if artifact.protected:
        return _protected_detail(artifact)
    if op.action == "delete":
        return _delete_refusal(op, artifact, plans, problem, inventory)
    return _write_refusal(name, artifact.digest, plans, problem)


def _delete_refusal(op: _Op, artifact: Any, plans: list[dict[str, Any]], problem: str,
                    inventory: Mapping[str, Any]) -> str | None:
    """A delete is authorized only by an exact absorbed-member binding in a validated plan."""
    name, umbrella = op.name, op.absorbed_into
    if not umbrella:
        return "delete must name the umbrella via absorbed_into"
    if umbrella == name:
        return "absorbed_into cannot equal the skill being deleted"
    if not plans:
        return _problem_detail(problem)
    absorbing = [plan for plan in plans if name in plan["absorbed"]]
    if not absorbing:
        return "no authorized merge plan absorbs this skill"
    matching = [plan for plan in absorbing if plan["canonical"] == umbrella]
    if not matching:
        expected = sorted({plan["canonical"] for plan in absorbing})
        if len(expected) == 1:
            return f"absorbed_into '{umbrella}' does not match the authorized plan (expected '{expected[0]}')"
        return "absorbed_into does not match any authorized plan"
    canonical = inventory.get(umbrella)
    if canonical is None:
        return f"the umbrella skill '{umbrella}' is not in the managed inventory"
    if canonical.protected:
        return _protected_detail(canonical)
    for plan in matching:
        if (plan["absorbed"][name] == artifact.digest
                and plan["canonical_digest"] == canonical.digest
                and pair_key(name, umbrella) in plan["edges"]):
            return None
    if not any(plan["absorbed"][name] == artifact.digest for plan in matching):
        return f"skill '{name}' changed after authorization (stale hash)"
    return f"the umbrella skill '{umbrella}' changed after authorization (stale hash)"


def _write_refusal(name: str, digest: str, plans: list[dict[str, Any]], problem: str) -> str | None:
    """Refuse content mutation: relation evidence authorizes absorption, never new bytes."""
    del digest, plans, problem
    return (f"skill '{name}' content mutation is not authorized by a relation plan; "
            "only exact hash-bound absorption is supported")


def _overwrites(artifact: Any, file_path: str) -> bool:
    """True when a write_file targets an existing file (or a target we cannot safely resolve)."""
    relative = str(file_path or "").strip()
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        return True  # malformed/escaping target: treat as destructive
    try:
        return (Path(artifact.path) / relative).is_file()
    except Exception:
        return True


def _block(op: _Op, detail: str) -> dict[str, str]:
    """Bounded, single-line block directive; also lands one audit row (never raises)."""
    action = _clip(op.action or "mutation", 32)
    name = _clip(op.name or "?", _MAX_NAME_CHARS)
    detail = _clip(detail, _MAX_DETAIL_CHARS)
    guidance = (" Install or refresh the authorized merge plan before retrying."
                if op.action == "delete" else "")
    message = _clip(
        f"jev-curator guard refused background skill_manage({action}) for '{name}': {detail}.{guidance}",
        _MAX_MESSAGE_CHARS,
    )
    try:
        audit("guard_block", action=action, skill=name, reason=detail)
    except Exception:
        pass
    return {"action": "block", "message": message}


def _problem_detail(problem: str) -> str:
    return ("the installed authorization is unreadable or malformed" if problem == "malformed"
            else "no authorized merge plan is installed")


def _protected_detail(artifact: Any) -> str:
    reasons = _clip(",".join(sorted(artifact.protected_reasons)), 120)
    return (f"skill '{artifact.name}' is protected ({reasons})" if reasons
            else f"skill '{artifact.name}' is protected")


# --- shape normalization ---------------------------------------------------------------

def _normalize_ops(args: Any) -> list[_Op]:
    """Flat (one op in ``args``) and ``operations`` batch shapes -> normalized ops."""
    if not isinstance(args, Mapping):
        return []
    batch = args.get("operations")
    if batch is None:
        return [_op_from(args, "")]
    if not isinstance(batch, (list, tuple)):
        return []  # malformed batch: core refuses it, nothing destructive we can name
    default_name = args.get("name")
    ops: list[_Op] = []
    for item in batch:
        op = _op_from(item, default_name)
        if op is not None:
            ops.append(op)
    return ops


def _op_from(item: Any, default_name: Any) -> _Op | None:
    if not isinstance(item, Mapping):
        return None
    action = _text(item.get("action")).lower()
    if not action:
        return None
    return _Op(action=action, name=_text(item.get("name")) or _text(default_name),
               file_path=_text(item.get("file_path")),
               absorbed_into=_text(item.get("absorbed_into")))


# --- plan store -------------------------------------------------------------------------

def _load_plan_entries() -> tuple[list[dict[str, Any]], str]:
    """(valid entries, problem) with problem in ``""``, ``"missing"``, ``"malformed"``."""
    path = plans_path()
    try:
        exists = path.is_file()
    except OSError:
        exists = False
    raw = read_json(path, None)
    if raw is None:
        return [], ("malformed" if exists else "missing")
    if not isinstance(raw, Mapping):
        return [], "malformed"
    entries = raw.get("plans")
    if entries is None or (isinstance(entries, Mapping) and not entries):
        return [], "missing"
    if not isinstance(entries, Mapping):
        return [], "malformed"
    valid = [entry for entry in (_valid_entry(value) for value in entries.values()) if entry]
    return (valid, "") if valid else ([], "malformed")


def _valid_entry(item: Any) -> dict[str, Any] | None:
    """Re-validate one stored entry; None when it is malformed or not installable."""
    stored_at = item.get("installed_at") if isinstance(item, Mapping) else None
    try:
        installed_at = float(stored_at) if stored_at is not None else 0.0
    except (TypeError, ValueError):
        installed_at = 0.0
    return _entry_from(item, installed_at=installed_at)


def _entry_from(plan: Any, *, installed_at: float | None = None) -> dict[str, Any] | None:
    """Normalize a MergePlan/mapping into a stored, hash-bound entry, or None when invalid."""
    if not isinstance(plan, Mapping):
        plan = {
            "plan_id": getattr(plan, "plan_id", ""),
            "status": getattr(plan, "status", ""),
            "blockers": getattr(plan, "blockers", ()),
            "canonical": getattr(plan, "canonical", ""),
            "canonical_digest": getattr(plan, "canonical_digest", ""),
            "absorbed_digests": getattr(plan, "absorbed_digests", {}),
            "relation_keys": getattr(plan, "relation_keys", ()),
        }
    plan_id = _text(plan.get("plan_id"))
    canonical = _text(plan.get("canonical"))
    canonical_digest = _text(plan.get("canonical_digest"))
    absorbed_raw = plan.get("absorbed_digests")
    if not isinstance(absorbed_raw, Mapping):
        absorbed_raw = plan.get("absorbed")
    absorbed = ({_text(key): _text(value) for key, value in absorbed_raw.items()}
                if isinstance(absorbed_raw, Mapping) else {})
    edges = tuple(sorted(_text(edge) for edge in (plan.get("relation_keys") or plan.get("edges") or ())
                         if _text(edge)))
    if not plan_id or len(plan_id) > _MAX_ID_CHARS or _text(plan.get("status")) != "validated":
        return None
    if plan.get("blockers"):
        return None
    if not canonical or not canonical_digest or not absorbed:
        return None
    if not all(absorbed.keys()) or not all(absorbed.values()) or canonical in absorbed:
        return None
    if not all(pair_key(member, canonical) in edges for member in absorbed):
        return None  # every absorbed member needs its own direct edge to the canonical
    return {
        "plan_id": plan_id,
        "status": "validated",
        "canonical": canonical,
        "canonical_digest": canonical_digest,
        "absorbed": absorbed,
        "edges": list(edges),
        "installed_at": time.time() if installed_at is None else installed_at,
    }


# --- environment seams -------------------------------------------------------------------

def _background_review() -> bool:
    """True inside the autonomous review fork; False on any lookup failure (mirrors core)."""
    try:
        from tools.skill_provenance import is_background_review
        return bool(is_background_review())
    except Exception:
        return False


def _active_mode(source: Any) -> str | None:
    """``guard``/``apply`` when the resolved settings arm the guard, else None (inert)."""
    try:
        value = source() if callable(source) else source
        if isinstance(value, Mapping):
            raw = value.get("mode")
        else:
            raw = getattr(getattr(value, "settings", value), "mode", None)
        mode = str(raw or "").strip().lower()
        return mode if mode in _ACTIVE_MODES else None
    except Exception:
        return None


def _current_artifacts(names: Iterable[str]) -> dict[str, Any] | None:
    """{name: SkillArtifact} for the requested names; None when the inventory is unreadable."""
    try:
        wanted = {str(name) for name in names if name}
        if not wanted:
            return {}
        found: dict[str, Any] = {}
        for item in collect_inventory(include_unmanaged=True, names=wanted):
            for alias in artifact_host_names(item.name, Path(item.path)):
                if alias not in wanted:
                    continue
                prior = found.get(alias)
                if prior is not None and Path(prior.path) != Path(item.path):
                    return None  # ambiguous host lookup: fail closed
                found[alias] = item
        return found
    except Exception:
        return None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clip(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]
