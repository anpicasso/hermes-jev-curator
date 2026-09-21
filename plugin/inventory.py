"""Safe, read-only inventory of curator-managed skill packages."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import SkillArtifact


_SUPPORT_DIRS = frozenset({"references", "templates", "scripts", "assets"})
_TEXT_SUFFIXES = frozenset({"", ".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".sh", ".ps1", ".html", ".css", ".sql"})
_MAX_TEXT_FILE_BYTES = 256_000
_MAX_PACKAGE_TEXT_BYTES = 1_000_000
_MAX_HASH_BYTES = 8_000_000


def collect_inventory(*, skills_root: Path | None = None, include_unmanaged: bool = False,
                      names: Iterable[str] | None = None) -> list[SkillArtifact]:
    """Collect local skill packages without following symlinks or reading binary/huge files."""
    from agent.skill_utils import (
        ESSENTIAL_SKILLS,
        get_disabled_skill_names,
        is_org_mirror_path,
        iter_skill_index_files,
        parse_frontmatter,
    )
    from hermes_constants import get_hermes_home
    from tools import skill_usage

    root = (skills_root or (Path(get_hermes_home()) / "skills")).expanduser().resolve()
    rows = skill_usage.usage_report() if include_unmanaged else skill_usage.curated_report()
    by_name: dict[str, Mapping[str, Any]] = {str(row.get("name") or ""): row for row in rows}
    cron_refs = _cron_references()
    disabled = get_disabled_skill_names()
    artifacts: list[SkillArtifact] = []
    seen: set[str] = set()
    wanted = {str(name) for name in names if str(name)} if names is not None else None

    if not root.exists():
        return []
    for skill_md in iter_skill_index_files(root, "SKILL.md"):
        try:
            if skill_md.is_symlink() or not skill_md.is_file():
                continue
            package = skill_md.parent.resolve(strict=True)
            if root not in package.parents and package != root:
                continue
            raw = skill_md.read_text(encoding="utf-8")
            frontmatter, _body = parse_frontmatter(raw)
            name = str((frontmatter or {}).get("name") or package.name).strip()
        except (OSError, UnicodeError, RuntimeError, ValueError):
            continue
        host_names = artifact_host_names(name, package, skills_root=root)
        usage_name = next((candidate for candidate in host_names if candidate in by_name), "")
        if (not name or name in seen or not usage_name
                or (wanted is not None and wanted.isdisjoint(host_names))):
            continue
        seen.add(name)
        row = by_name[usage_name]
        text, support_files, digest, package_flags = _read_package(package)
        reasons = set(package_flags)
        if package.name != name:
            reasons.add("name-path-mismatch")
        provenance = str(row.get("provenance") or skill_usage.provenance(usage_name))
        if bool(row.get("pinned")):
            reasons.add("pinned")
        if provenance in {"bundled", "hub", "external"}:
            reasons.add(provenance)
        if name in disabled:
            reasons.add("disabled")
        if name in ESSENTIAL_SKILLS:
            reasons.add("essential")
        if is_org_mirror_path(package, root):
            reasons.add("org-mirror")
        if name in cron_refs:
            reasons.add("cron-referenced")
        try:
            if not skill_usage.is_curation_eligible(usage_name, package):
                reasons.add("core-ineligible")
        except Exception:
            reasons.add("eligibility-unknown")
        artifacts.append(SkillArtifact(
            name=name,
            path=package,
            description=str((frontmatter or {}).get("description") or "").strip(),
            text=text,
            digest=digest,
            provenance=provenance,
            state=str(row.get("state") or "active"),
            pinned=bool(row.get("pinned")),
            use_count=_safe_int(row.get("use_count")),
            last_activity_at=str(row.get("last_activity_at") or ""),
            support_files=tuple(support_files),
            protected_reasons=tuple(sorted(reasons)),
        ))
    return sorted(artifacts, key=lambda artifact: artifact.name)


def artifact_host_names(name: str, package: Path, *,
                        skills_root: Path | None = None) -> tuple[str, ...]:
    """Names accepted by the host for a package, plus its frontmatter name.

    ``skill_manage`` resolves a bare directory name and, for local categorized
    skills, the path relative to ``skills/``.  Keeping those aliases here lets
    callers fail closed when frontmatter and path names disagree.
    """
    if skills_root is None:
        from hermes_constants import get_hermes_home
        skills_root = Path(get_hermes_home()) / "skills"
    candidates = [str(name or "").strip(), package.name]
    try:
        relative = package.resolve().relative_to(skills_root.expanduser().resolve()).as_posix()
        candidates.append(relative)
    except (OSError, RuntimeError, ValueError):
        pass
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def artifact_by_name(name: str, *, skills_root: Path | None = None) -> SkillArtifact | None:
    return next(iter(collect_inventory(
        skills_root=skills_root, include_unmanaged=True, names={name})), None)


def digests_match(expected: Mapping[str, str], inventory: Iterable[SkillArtifact]) -> bool:
    actual = {item.name: item.digest for item in inventory}
    return bool(expected) and all(
        isinstance(digest, str) and bool(digest) and name in actual and actual[name] == digest
        for name, digest in expected.items()
    )


def _read_package(package: Path) -> tuple[str, list[str], str, tuple[str, ...]]:
    hasher = hashlib.sha256()
    chunks: list[str] = []
    support: list[str] = []
    flags: set[str] = set()
    text_bytes = 0
    hash_bytes = 0

    try:
        paths = sorted(package.rglob("*"), key=lambda path: path.as_posix())
    except OSError:
        return "", [], hashlib.sha256(b"unreadable").hexdigest(), ("unreadable-package",)

    for path in paths:
        try:
            relative = path.relative_to(package).as_posix()
            stat = path.lstat()
        except (OSError, ValueError):
            flags.add("unreadable-entry")
            continue
        if path.is_symlink():
            flags.add("contains-symlink")
            continue
        if not path.is_file():
            continue
        hasher.update(relative.encode("utf-8", "surrogateescape") + b"\0")
        hasher.update(str(stat.st_size).encode() + b"\0")
        if hash_bytes + stat.st_size > _MAX_HASH_BYTES:
            flags.add("package-too-large")
            continue
        try:
            data = path.read_bytes()
        except OSError:
            flags.add("unreadable-entry")
            continue
        hash_bytes += len(data)
        hasher.update(data)
        top = relative.split("/", 1)[0]
        if relative != "SKILL.md" and top not in _SUPPORT_DIRS:
            continue
        if relative != "SKILL.md":
            support.append(relative)
        if path.suffix.lower() not in _TEXT_SUFFIXES or b"\0" in data[:4096]:
            flags.add("contains-binary")
            continue
        if len(data) > _MAX_TEXT_FILE_BYTES or text_bytes + len(data) > _MAX_PACKAGE_TEXT_BYTES:
            flags.add("text-too-large")
            continue
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError:
            flags.add("contains-non-utf8")
            continue
        text_bytes += len(data)
        chunks.append(f"\n\n===== {relative} =====\n{decoded}")
    return "".join(chunks).lstrip(), support, hasher.hexdigest(), tuple(sorted(flags))


def _cron_references() -> set[str]:
    try:
        from cron.jobs import referenced_skill_names
        return set(referenced_skill_names())
    except Exception:
        return set()


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
