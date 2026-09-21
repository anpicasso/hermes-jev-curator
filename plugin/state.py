"""Durable, profile-local storage for the Jev curator.

Every byte this plugin persists lives under one directory inside the *active* Hermes
home -- ``~/.hermes/jev-curator``, or ``~/.hermes/profiles/<name>/jev-curator`` for a
profile -- and never inside ``skills/``: the core-owned ``skills/.usage.json`` and
``skills/.curator_state`` stay untouchable, so even a crashed curator cannot corrupt
core bookkeeping.

    <root>/audit.jsonl     append-only redacted JSONL trail, one .1 generation, 0600
    <root>/state.json      last-run durable state, atomic replace, 0600
    <root>/relations.json  content-hash relation cache, atomic replace, 0600
    <root>/claim.lock      O_EXCL single-claim lock with serialized stale recovery, 0600
    <root>/reports/*.json  run reports, atomic replace, 0600

The root always follows ``hermes_constants.get_hermes_home()`` so state cannot cross profile
boundaries. ``JEV_CURATOR_AUDIT_MAX_BYTES`` bounds the audit log (0 disables it).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping

from .models import RelationJudgment

_APP_DIR = "jev-curator"
_AUDIT_MAX_BYTES = 2_000_000
_LOCK_STALE_SECONDS = 1800.0
_CACHE_MAX_ENTRIES = 2_000
_CORE_FILE_NAMES = frozenset({".usage.json", ".curator_state"})
_MAX_FIELD_CHARS = 600
_MAX_ITEMS = 200
_MAX_DEPTH = 6
_MAX_JSON_BYTES = 10_000_000
_REPORT_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
_JUDGMENT_FIELDS = tuple(field.name for field in fields(RelationJudgment))


# --- paths ---------------------------------------------------------------------------

def state_root() -> Path:
    """Active profile's curator directory."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / _APP_DIR
    except Exception:
        configured = (os.environ.get("HERMES_HOME") or "").strip()
        base = Path(configured).expanduser() if configured else Path.home() / ".hermes"
        return base / _APP_DIR


def audit_log_path() -> Path:
    return state_root() / "audit.jsonl"


def state_path() -> Path:
    return state_root() / "state.json"


def relations_path() -> Path:
    return state_root() / "relations.json"


def lock_path() -> Path:
    return state_root() / "claim.lock"


def reports_dir() -> Path:
    return state_root() / "reports"


# --- atomic JSON ---------------------------------------------------------------------

def write_json(path: Path | str, payload: Any) -> Path:
    """Serialize to JSON and swap it in atomically (same-dir temp + ``os.replace``).

    ponytail: file fsync + rename is the atomicity contract; no directory fsync, add
    one only if power-loss durability of the rename itself matters.
    """
    target = _guarded(Path(path))
    if target.is_symlink():
        raise ValueError("refusing to replace a symlinked state file")
    _ensure_dir(target.parent)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def read_json(path: Path | str, default: Any = None) -> Any:
    """Parse bounded regular JSON without following a final symlink."""
    fd = -1
    try:
        flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        fd = os.open(Path(path), flags)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_JSON_BYTES:
            os.close(fd)
            fd = -1
            return default
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            return json.load(handle)
    except (OSError, ValueError, UnicodeError):
        return default
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def load_state() -> dict[str, Any]:
    data = read_json(state_path(), {})
    return data if isinstance(data, dict) else {}


def save_state(payload: Mapping[str, Any]) -> Path:
    return write_json(state_path(), dict(payload))


# --- content-hash relation cache -----------------------------------------------------

def relation_key(a_digest: str, b_digest: str, *, contract_version: str, model: str = "") -> str:
    """Cache key over the two content digests, the question contract, and the model.

    Digest order does not matter (``a``/``b`` are symmetric); a changed skill body, a
    changed contract, or a changed judge model all miss the cache. Pass the same
    ``model`` on lookup and store; ``""`` means model-agnostic.
    """
    payload = "::".join(sorted((str(a_digest), str(b_digest))) + [str(contract_version), str(model)])
    return hashlib.sha256(payload.encode("utf-8", "surrogateescape")).hexdigest()[:32]


def load_relation_cache() -> dict[str, Any]:
    data = read_json(relations_path(), {})
    entries = data.get("entries") if isinstance(data, Mapping) else None
    if not isinstance(entries, Mapping):
        return {}
    return {str(key): dict(value) for key, value in entries.items() if isinstance(value, Mapping)}


def save_relation_cache(cache: Mapping[str, Any], *, max_entries: int | None = None) -> Path:
    entries = {str(key): value for key, value in cache.items() if isinstance(value, Mapping)}
    cap = _CACHE_MAX_ENTRIES if max_entries is None else int(max_entries)
    if cap > 0 and len(entries) > cap:
        entries = dict(list(entries.items())[-cap:])  # insertion order: newest live last
    return write_json(relations_path(), {"version": 1, "entries": entries})


def cached_relation(
    cache: Mapping[str, Any], a_digest: str, b_digest: str, *, contract_version: str, model: str = "",
) -> RelationJudgment | None:
    entry = cache.get(relation_key(a_digest, b_digest, contract_version=contract_version, model=model))
    raw = entry.get("judgment") if isinstance(entry, Mapping) else None
    return judgment_from_dict(raw)


def remember_relation(
    cache: MutableMapping[str, Any], judgment: RelationJudgment, *, model: str = "",
    stored_at: float | None = None,
) -> str:
    """Insert/re-insert a judgment (moving it to newest) and return its cache key."""
    key = relation_key(judgment.a_digest, judgment.b_digest,
                       contract_version=judgment.contract_version, model=model)
    cache.pop(key, None)
    cache[key] = {
        "stored_at": time.time() if stored_at is None else float(stored_at),
        "judgment": judgment_to_dict(judgment),
    }
    return key


def judgment_to_dict(judgment: RelationJudgment) -> dict[str, Any]:
    return asdict(judgment)


def judgment_from_dict(raw: Any) -> RelationJudgment | None:
    """Rebuild a judgment from cache; ``None`` when the row is malformed or invalid."""
    if not isinstance(raw, Mapping):
        return None
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
           for value in probabilities.values()):
        return None
    data = {name: raw[name] for name in _JUDGMENT_FIELDS if name in raw}
    try:
        return RelationJudgment(**data)
    except (TypeError, ValueError):
        return None


# --- claim lock ----------------------------------------------------------------------

@contextmanager
def claim_lock(*, stale_seconds: float = _LOCK_STALE_SECONDS) -> Iterator[bool]:
    """Yield True when this process holds the single curator claim, False when busy.

    ``os.O_EXCL`` creation is the arbitration. Recovery of a dead-owner or old
    malformed claim is serialized with a kernel lock so two recoverers cannot both
    delete each other's newly acquired claim.
    """
    path = lock_path()
    try:
        _ensure_dir(path.parent)
    except (OSError, ValueError):
        yield False
        return
    acquired = _try_acquire(path)
    if not acquired:
        acquired = _recover_stale(path, stale_seconds)
    try:
        yield acquired
    finally:
        if acquired:
            _release(path, only_if_pid=os.getpid())


def _try_acquire(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError:
        return False
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "at": time.time()}).encode("utf-8"))
    except OSError:
        pass  # an empty lock still blocks others and is stale-recoverable
    finally:
        os.close(fd)
    return True


def _recover_stale(path: Path, stale_seconds: float) -> bool:
    """Serialize stale-claim replacement; fail closed without advisory locks."""
    try:
        import fcntl
    except ImportError:
        return False
    guard = path.with_name(".claim-recovery.lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(guard, flags, 0o600)
    except OSError:
        return False
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not _stale(path, stale_seconds):
            return False
        _release(path)
        return _try_acquire(path)
    except OSError:
        return False
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _stale(path: Path, stale_seconds: float) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False  # vanished; the retry's O_EXCL decides
    pid = _lock_pid(path)
    if pid is not None:
        return not _pid_alive(pid)
    return age > max(1.0, float(stale_seconds))


def _lock_pid(path: Path) -> int | None:
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
            return None
        raw = os.read(fd, 4097)
        if len(raw) > 4096:
            return None
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    finally:
        os.close(fd)
    pid = data.get("pid") if isinstance(data, Mapping) else None
    return int(pid) if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # EPERM et al: the pid exists
    return True


def _release(path: Path, *, only_if_pid: int | None = None) -> None:
    try:
        if only_if_pid is not None and _lock_pid(path) not in (None, only_if_pid):
            return  # a stale-recoverer already handed the claim to someone else
        path.unlink()
    except OSError:
        pass


# --- reports -------------------------------------------------------------------------

def write_report(name: str, payload: Any) -> Path:
    data = asdict(payload) if is_dataclass(payload) and not isinstance(payload, type) else payload
    return write_json(_report_path(name), _safe(data, 0))


def read_report(name: str) -> dict[str, Any] | None:
    data = read_json(_report_path(name))
    return data if isinstance(data, dict) else None


def list_reports() -> list[str]:
    try:
        return sorted(entry.stem for entry in reports_dir().glob("*.json")
                      if entry.is_file() and not entry.is_symlink())
    except OSError:
        return []


def _report_path(name: str) -> Path:
    cleaned = str(name or "").strip()
    if not _REPORT_NAME_RE.fullmatch(cleaned) or cleaned.startswith(".") or ".." in cleaned:
        raise ValueError("report name must be a simple file name (letters, digits, . _ -)")
    return reports_dir() / f"{cleaned}.json"


# --- audit ---------------------------------------------------------------------------

def audit(event: str, **fields: Any) -> bool:
    """Append one redacted, bounded JSONL row. Never raises; True when persisted.

    ponytail: one O_APPEND write is enough for a diagnostic trail -- no lock, no
    fsync. Rotation keeps exactly one generation.
    """
    try:
        max_bytes = _audit_max_bytes()
        if max_bytes <= 0:
            return False
        path = audit_log_path()
        _ensure_dir(path.parent)
        row = {**_safe(fields, 0), "ts": time.time(), "event": _redact(str(event))[:120]}
        encoded = (json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":")) + "\n").encode("utf-8")
        if len(encoded) > max_bytes:
            return False
        try:
            if path.stat().st_size and path.stat().st_size + len(encoded) > max_bytes:
                os.replace(path, path.with_suffix(path.suffix + ".1"))
        except FileNotFoundError:
            pass
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            os.write(fd, encoded)
        finally:
            os.close(fd)
        return True
    except Exception:
        return False


def _audit_max_bytes() -> int:
    raw = (os.environ.get("JEV_CURATOR_AUDIT_MAX_BYTES") or "").strip()
    if not raw:
        return _AUDIT_MAX_BYTES
    try:
        return int(raw)
    except ValueError:
        return _AUDIT_MAX_BYTES


def _safe(value: Any, depth: int) -> Any:
    """Bound and redact one audit value; depth/width caps stop hostile nesting."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact(value[:_MAX_FIELD_CHARS * 2])[:_MAX_FIELD_CHARS]
    if isinstance(value, Mapping):
        if depth >= _MAX_DEPTH:
            return "<depth-limit>"
        return {_redact(str(key))[:80]: _safe(item, depth + 1)
                for key, item in list(value.items())[:_MAX_ITEMS]}
    if isinstance(value, (list, tuple, set, frozenset)):
        if depth >= _MAX_DEPTH:
            return "<depth-limit>"
        return [_safe(item, depth + 1) for item in list(value)[:_MAX_ITEMS]]
    return _redact(str(value)[:_MAX_FIELD_CHARS * 2])[:_MAX_FIELD_CHARS]


_FALLBACK_SECRET_RE = re.compile(
    r"(?i:\b(?:sk-[A-Za-z0-9_.-]{9,}|gh[pousr]_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[A-Za-z0-9_-]{30,}"
    r"|bearer\s+[A-Za-z0-9._~+/-]{8,}=*"
    r"|(?:api[_-]?key|access[_-]?token|token|secret|password)\b\s*[:=]\s*[^\s,;\"'}\]]{4,}))"
    r"|\b[A-Z][A-Z0-9_]{2,40}\s*[:=]\s*[^\s,;\"'}\]]{4,}",  # ENV_STYLE=... assignments
)


def _redact(text: str) -> str:
    """Canonical Hermes redactor plus the plugin's shell/HTTP credential rules."""
    locally_redacted = _FALLBACK_SECRET_RE.sub("[redacted]", text)
    try:
        from .transport import _local_redact

        locally_redacted = _local_redact(locally_redacted)
    except Exception:
        pass  # audit is fail-soft; the independent broad fallback already ran
    try:
        from agent.redact import redact_sensitive_text

        try:
            return redact_sensitive_text(locally_redacted, force=True, redact_url_credentials=True)
        except TypeError:  # older signature without the URL-credential flag
            return redact_sensitive_text(locally_redacted, force=True)
    except Exception:
        return locally_redacted


def redact_text(text: str) -> str:
    """Public fail-soft redaction choke point for outbound and persisted free text."""
    return _redact(str(text))


# --- internals -----------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("refusing a symlinked state directory")
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not path.is_dir():
        raise ValueError("state path is not a directory")
    os.chmod(path, 0o700)
    return path


def _guarded(path: Path) -> Path:
    if path.name in _CORE_FILE_NAMES:
        raise ValueError(f"refusing to write core-owned {path.name!r}")
    root = state_root()
    if root.is_symlink():
        raise ValueError("refusing a symlinked state root")
    resolved_root = root.resolve(strict=False)
    resolved_parent = path.parent.resolve(strict=False)
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise ValueError("state writes must remain under the active profile's jev-curator directory")
    return path
