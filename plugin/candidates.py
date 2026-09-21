"""Deterministic candidate generation; Jev only sees the ambiguous residue."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from .models import CandidatePair, SkillArtifact


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.-]*", re.I)
_MIN_SIMILARITY = 0.12


def generate_candidates(
    artifacts: Iterable[SkillArtifact], *, top_k: int = 5, max_pairs: int = 100,
) -> list[CandidatePair]:
    """Return stable top-k lexical neighbors per skill, deduplicated and budgeted."""
    # Core's consolidation fork cannot read disabled/protected skills. Keep
    # them visible in inventory reports, but never propose them for mutation.
    items = sorted(
        (item for item in artifacts if not item.protected and item.state != "archived"),
        key=lambda item: item.name,
    )
    features = {item.name: _features(item) for item in items}
    neighbors: dict[str, list[tuple[float, SkillArtifact, tuple[str, ...]]]] = defaultdict(list)

    for index, a in enumerate(items):
        for b in items[index + 1:]:
            score, signals = _similarity(features[a.name], features[b.name])
            if score >= _MIN_SIMILARITY or "name-prefix" in signals:
                neighbors[a.name].append((score, b, signals))
                neighbors[b.name].append((score, a, signals))

    selected: dict[str, CandidatePair] = {}
    by_name = {item.name: item for item in items}
    for name in sorted(neighbors):
        ranked = sorted(neighbors[name], key=lambda row: (-row[0], row[1].name))[:max(1, top_k)]
        for score, other, signals in ranked:
            a, b = sorted((by_name[name], other), key=lambda item: item.name)
            pair = CandidatePair(
                a=a.name,
                b=b.name,
                a_digest=a.digest,
                b_digest=b.digest,
                similarity=round(score, 6),
                signals=signals,
            )
            selected[pair.key] = pair
    return sorted(selected.values(), key=lambda pair: (-pair.similarity, pair.a, pair.b))[:max_pairs]


def deterministic_relation(pair: CandidatePair) -> str:
    """Cheap ablation baseline. It proposes only; it never authorizes a mutation."""
    if pair.similarity >= 0.92:
        return "duplicate"
    if pair.similarity >= 0.72:
        return "same_class"
    return "unrelated"


def _features(item: SkillArtifact) -> dict[str, Any]:
    name_tokens = _tokens(item.name.replace("-", " "))
    description_tokens = _tokens(item.description)
    body_tokens = _tokens(item.text)
    return {
        "name": name_tokens,
        "description": description_tokens,
        "body": body_tokens,
        "counts": Counter(body_tokens + description_tokens * 2 + name_tokens * 3),
        "prefix": item.name.split("-", 1)[0].lower(),
    }


def _similarity(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, tuple[str, ...]]:
    name_j = _jaccard(set(a["name"]), set(b["name"]))
    desc_j = _jaccard(set(a["description"]), set(b["description"]))
    cosine = _cosine(a["counts"], b["counts"])
    same_prefix = bool(a["prefix"] and a["prefix"] == b["prefix"])
    score = min(1.0, 0.55 * cosine + 0.25 * desc_j + 0.15 * name_j + (0.05 if same_prefix else 0.0))
    signals = []
    if same_prefix:
        signals.append("name-prefix")
    if name_j >= 0.5:
        signals.append("name-overlap")
    if desc_j >= 0.35:
        signals.append("description-overlap")
    if cosine >= 0.45:
        signals.append("content-overlap")
    return score, tuple(signals)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) > 1]


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
