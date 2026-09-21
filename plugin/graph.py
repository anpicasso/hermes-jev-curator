"""Deterministic relationship graph and merge-plan builder.

Absorption is star-shaped, never transitive: every absorbed member carries its own
direct high-confidence containment or duplicate judgment against the plan's
canonical, so A~B plus B~C never merges C into A. Merge plan IDs are content
hashes, and the evidence policy is versioned in code, not operator-configurable.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .models import MergePlan, RelationJudgment, SkillArtifact


# Versioned evidence policy. Tune only with measurements, never from settings.
GRAPH_VERSION = "skill-graph-v1"
MIN_CONFIDENCE = 0.85      # confidence in the absorbing relation choice
MIN_COVERAGE = 0.70        # the offered options adequately describe the pair
MAX_CONFLICT = 0.15        # tolerated operational conflict between two skills
MIN_PRESERVATION = 0.90    # absorbed content must survive in the canonical
MAX_PLANS = 20
ABSORBING_RELATIONS = frozenset({"duplicate", "a_subset_of_b", "b_subset_of_a"})


def pair_key(a: str, b: str) -> str:
    """Canonical pair key; matches CandidatePair.key and RelationJudgment.key."""
    return "::".join(sorted((str(a), str(b))))


@dataclass(frozen=True)
class Edge:
    """One deduplicated judgment. `authorized` is the only absorption gate."""

    key: str
    a: str
    b: str
    relation: str
    confidence: float
    coverage: float
    conflict: float
    preservation_a_in_b: float
    preservation_b_in_a: float
    authorized: bool = False
    canonical: str = ""
    absorbed: str = ""
    refusals: tuple[str, ...] = ()

    @property
    def conflicted(self) -> bool:
        """Operational conflict at any confidence: such pairs must never co-merge."""
        return self.relation == "conflict" or self.conflict > MAX_CONFLICT


@dataclass(frozen=True)
class RelationshipGraph:
    """Evaluated judgments over the inventory; the input to every merge plan."""

    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    version: str = GRAPH_VERSION

    def edge(self, a: str, b: str) -> Edge | None:
        key = pair_key(a, b)
        return next((edge for edge in self.edges if edge.key == key), None)

    @property
    def authorized(self) -> tuple[Edge, ...]:
        return tuple(edge for edge in self.edges if edge.authorized)

    def refusal_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edges:
            for reason in edge.refusals:
                counts[reason] = counts.get(reason, 0) + 1
        return counts


def build_graph(
    artifacts: Iterable[SkillArtifact],
    judgments: Iterable[RelationJudgment],
    *,
    truncated_pairs: Iterable[str] = (),
) -> RelationshipGraph:
    """Evaluate each judgment once, in stable order, without mutating anything.

    `truncated_pairs` carries pair keys whose prompt state was explicitly truncated
    (the engine instead downgrades those to `insufficient_evidence`); either way
    truncated evidence can never authorize absorption.
    """
    items = {item.name: item for item in artifacts}
    truncated = {str(key) for key in truncated_pairs}
    best: dict[str, RelationJudgment] = {}
    for judgment in sorted(judgments, key=lambda item: (item.key, -item.confidence, -item.coverage, item.relation)):
        best.setdefault(judgment.key, judgment)
    edges = tuple(_evaluate(best[key], items, truncated) for key in sorted(best))
    return RelationshipGraph(nodes=tuple(sorted(items)), edges=edges)


def build_plans(
    artifacts: Iterable[SkillArtifact],
    judgments: Iterable[RelationJudgment],
    *,
    max_plans: int = MAX_PLANS,
    truncated_pairs: Iterable[str] = (),
) -> list[MergePlan]:
    """Build validated/blocked star merges, or one explicit no-op plan."""
    items = {item.name: item for item in artifacts}
    graph = build_graph(items.values(), judgments, truncated_pairs=truncated_pairs)

    claims: dict[str, dict[str, Edge]] = {}
    for edge in graph.authorized:
        claims.setdefault(edge.canonical, {})[edge.absorbed] = edge

    # A member may satisfy the direct-edge rule for several canonicals (three
    # duplicate skills); the best-ranked claimant owns it so no skill is deleted
    # by two plans. ponytail: rank order, revisit if merge order ever matters.
    canonicals = sorted(claims, key=lambda name: _rank(items[name]))
    owner: dict[str, str] = {}
    for canonical in canonicals:
        for member in sorted(claims[canonical]):
            owner.setdefault(member, canonical)

    plans: list[MergePlan] = []
    for canonical in canonicals:
        edges = {member: edge for member, edge in claims[canonical].items() if owner[member] == canonical}
        if not edges:
            continue
        absorbed = tuple(sorted(edges))
        blockers = list(_group_blockers(graph, items, canonical, absorbed))
        if canonical in owner:
            blockers.append(f"canonical-absorbed-by:{owner[canonical]}")
        plans.append(_plan(canonical, items, edges, absorbed, blockers))

    plans.sort(key=lambda plan: (plan.status != "validated", -len(plan.absorbed), plan.canonical))
    limit = max(1, int(max_plans))
    return plans[:limit] if plans else [_noop_plan(graph, items)]


def _evaluate(judgment: RelationJudgment, items: Mapping[str, SkillArtifact], truncated: set[str]) -> Edge:
    key = judgment.key
    refusals: list[str] = []
    if judgment.a == judgment.b:
        refusals.append("self-pair")
    if judgment.a not in items or judgment.b not in items:
        refusals.append("unknown-artifact")
    elif (items[judgment.a].digest != judgment.a_digest
          or items[judgment.b].digest != judgment.b_digest):
        refusals.append("stale-hash")
    if key in truncated:
        refusals.append("truncated")
    if not _units(judgment):
        refusals.append("malformed-evidence")

    authorized = False
    canonical = absorbed = ""
    if judgment.relation == "insufficient_evidence":
        refusals.append("insufficient-evidence")
    elif judgment.relation == "conflict":
        refusals.append("conflict")
    elif judgment.relation in ABSORBING_RELATIONS:
        if judgment.confidence < MIN_CONFIDENCE:
            refusals.append("low-confidence")
        if judgment.coverage < MIN_COVERAGE:
            refusals.append("low-coverage")
        if judgment.conflict > MAX_CONFLICT:
            refusals.append("conflict-score")
        if not refusals:
            canonical, absorbed, preservation = _direction(judgment, items)
            if preservation < MIN_PRESERVATION:
                refusals.append("low-preservation")
            else:
                authorized = True

    return Edge(
        key=key, a=judgment.a, b=judgment.b, relation=judgment.relation,
        confidence=float(judgment.confidence), coverage=float(judgment.coverage),
        conflict=float(judgment.conflict),
        preservation_a_in_b=float(judgment.preservation_a_in_b),
        preservation_b_in_a=float(judgment.preservation_b_in_a),
        authorized=authorized, canonical=canonical, absorbed=absorbed,
        refusals=tuple(refusals),
    )


def _direction(judgment: RelationJudgment, items: Mapping[str, SkillArtifact]) -> tuple[str, str, float]:
    """(canonical, absorbed, preservation of absorbed inside canonical)."""
    if judgment.relation == "a_subset_of_b":
        return judgment.b, judgment.a, judgment.preservation_a_in_b
    if judgment.relation == "b_subset_of_a":
        return judgment.a, judgment.b, judgment.preservation_b_in_a
    canonical, absorbed = sorted((judgment.a, judgment.b), key=lambda name: _rank(items[name]))
    return canonical, absorbed, min(judgment.preservation_a_in_b, judgment.preservation_b_in_a)


def _rank(item: SkillArtifact) -> tuple[int, str]:
    """Deterministic canonical preference: most-used skill first, then name."""
    return (-int(item.use_count or 0), item.name)


def _group_blockers(
    graph: RelationshipGraph, items: Mapping[str, SkillArtifact], canonical: str, absorbed: tuple[str, ...],
) -> tuple[str, ...]:
    blockers: list[str] = []
    members = (canonical, *absorbed)
    for name in members:
        item = items[name]
        if item.protected:
            reasons = ",".join(item.protected_reasons)
            blockers.append(f"protected:{name}:{reasons}" if reasons else f"protected:{name}")
    for index, first in enumerate(members):
        for second in members[index + 1:]:
            edge = graph.edge(first, second)
            if edge is not None and edge.conflicted:
                blockers.append(f"conflict:{pair_key(first, second)}")
    return tuple(blockers)


def _plan(
    canonical: str, items: Mapping[str, SkillArtifact], edges: Mapping[str, Edge],
    absorbed: tuple[str, ...], blockers: list[str],
) -> MergePlan:
    absorbed_digests = {name: items[name].digest for name in absorbed}
    relation_keys = tuple(sorted(edge.key for edge in edges.values()))
    return MergePlan(
        plan_id=_plan_id(canonical, items[canonical].digest, absorbed_digests, relation_keys),
        canonical=canonical,
        canonical_digest=items[canonical].digest,
        absorbed=absorbed,
        absorbed_digests=absorbed_digests,
        relation_keys=relation_keys,
        status="blocked" if blockers else "validated",
        blockers=tuple(blockers),
        metadata={
            "graph_version": GRAPH_VERSION,
            "absorbed_count": len(absorbed),
            "relations": {edge.key: edge.relation for edge in sorted(edges.values(), key=lambda edge: edge.key)},
        },
    )


def _plan_id(canonical: str, canonical_digest: str, absorbed_digests: Mapping[str, str],
             relation_keys: tuple[str, ...]) -> str:
    material = "\n".join((
        GRAPH_VERSION, canonical, canonical_digest,
        *(f"{name}={absorbed_digests[name]}" for name in sorted(absorbed_digests)),
        *relation_keys,
    ))
    return "merge-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def _noop_plan(graph: RelationshipGraph, items: Mapping[str, SkillArtifact]) -> MergePlan:
    material = "\n".join((
        GRAPH_VERSION, "noop",
        *(f"{name}={items[name].digest}" for name in graph.nodes),
        *(f"{edge.key}:{edge.relation}" for edge in graph.edges),
    ))
    return MergePlan(
        plan_id="noop-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12],
        canonical="",
        canonical_digest="",
        absorbed=(),
        absorbed_digests={},
        relation_keys=(),
        status="noop",
        blockers=(),
        metadata={
            "graph_version": GRAPH_VERSION,
            "reason": "no-authorized-relations",
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "refusals": graph.refusal_counts(),
        },
    )


def _units(judgment: RelationJudgment) -> bool:
    return all(_is_unit(value) for value in (
        judgment.confidence, judgment.coverage, judgment.conflict,
        judgment.preservation_a_in_b, judgment.preservation_b_in_a,
    ))


def _is_unit(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0
