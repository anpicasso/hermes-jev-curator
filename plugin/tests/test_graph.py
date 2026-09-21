from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from plugin.graph import (
    GRAPH_VERSION,
    Edge,
    MAX_CONFLICT,
    MIN_COVERAGE,
    MIN_CONFIDENCE,
    build_graph,
    build_plans,
    pair_key,
)
from plugin.models import MergePlan, RelationJudgment, SkillArtifact


def artifact(
    name: str, *, digest: str = "", use_count: int = 0, protected: tuple[str, ...] = (),
) -> SkillArtifact:
    return SkillArtifact(
        name=name,
        path=Path("/skills") / name,
        description=f"does {name}",
        text=f"body for {name}",
        digest=digest or f"digest-{name}",
        use_count=use_count,
        protected_reasons=protected,
    )


def judgment(
    a: str, b: str, relation: str, *, confidence: float = 0.95, coverage: float = 0.9,
    conflict: float = 0.02, preservation_a_in_b: float = 0.95, preservation_b_in_a: float = 0.95,
    digest_a: str | None = None, digest_b: str | None = None,
) -> RelationJudgment:
    return RelationJudgment(
        a=a, b=b,
        a_digest=digest_a if digest_a is not None else f"digest-{a}",
        b_digest=digest_b if digest_b is not None else f"digest-{b}",
        relation=relation,
        confidence=confidence,
        probabilities={relation: 1.0},
        coverage=coverage,
        preservation_a_in_b=preservation_a_in_b,
        preservation_b_in_a=preservation_b_in_a,
        conflict=conflict,
        contract_version="skill-relations-v1",
    )


def require_edge(graph, a: str, b: str) -> Edge:
    edge = graph.edge(a, b)
    if edge is None:
        raise AssertionError(f"missing graph edge {pair_key(a, b)}")
    return edge


class MergeGraphTests(unittest.TestCase):
    def assert_star_shaped(self, graph, plans: list[MergePlan]) -> None:
        """Every absorbed member: one direct authorized edge, claimed by one plan."""
        seen: set[str] = set()
        for plan in plans:
            for member in plan.absorbed:
                edge = require_edge(graph, plan.canonical, member)
                self.assertTrue(edge.authorized)
                self.assertNotIn(member, seen, f"{member} is absorbed by two plans")
                seen.add(member)

    def test_duplicate_pair_builds_validated_star_plan(self):
        skills = [artifact("alpha", use_count=7), artifact("beta", use_count=1)]
        plans = build_plans(skills, [judgment("alpha", "beta", "duplicate")])
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.status, "validated")
        self.assertTrue(plan.applicable)
        self.assertEqual((plan.canonical, plan.absorbed), ("alpha", ("beta",)))
        self.assertEqual(plan.canonical_digest, "digest-alpha")
        self.assertEqual(dict(plan.absorbed_digests), {"beta": "digest-beta"})
        self.assertEqual(plan.relation_keys, ("alpha::beta",))
        self.assertEqual(plan.metadata["graph_version"], GRAPH_VERSION)
        self.assert_star_shaped(build_graph(skills, [judgment("alpha", "beta", "duplicate")]), plans)

    def test_subset_direction_decides_the_canonical(self):
        skills = [artifact("a"), artifact("b")]
        up = build_plans(skills, [judgment("a", "b", "a_subset_of_b")])[0]
        self.assertEqual((up.canonical, up.absorbed), ("b", ("a",)))
        down = build_plans(skills, [judgment("a", "b", "b_subset_of_a")])[0]
        self.assertEqual((down.canonical, down.absorbed), ("a", ("b",)))

    def test_non_absorbing_relations_never_merge(self):
        skills = [artifact("a"), artifact("b")]
        for relation in ("same_class", "complementary", "unrelated"):
            plans = build_plans(skills, [judgment("a", "b", relation)])
            self.assertEqual([plan.status for plan in plans], ["noop"], relation)

    def test_duplicate_canonical_is_deterministic(self):
        used = [artifact("a", use_count=0), artifact("b", use_count=4)]
        plan = build_plans(used, [judgment("a", "b", "duplicate")])[0]
        self.assertEqual((plan.canonical, plan.absorbed), ("b", ("a",)))
        tied = build_plans([artifact("a"), artifact("b")], [judgment("a", "b", "duplicate")])[0]
        self.assertEqual((tied.canonical, tied.absorbed), ("a", ("b",)))

    def test_plan_ids_are_stable_and_content_addressed(self):
        skills = [artifact("a"), artifact("b")]
        evidence = [judgment("a", "b", "duplicate")]
        first = build_plans(skills, evidence)
        self.assertEqual(first, build_plans(list(reversed(skills)), list(reversed(evidence))))
        self.assertTrue(first[0].plan_id.startswith("merge-"))
        moved = build_plans(
            [artifact("a"), artifact("b", digest="moved")],
            [judgment("a", "b", "duplicate", digest_b="moved")],
        )
        self.assertNotEqual(first[0].plan_id, moved[0].plan_id)
        self.assertEqual(moved[0].absorbed_digests, {"b": "moved"})

    def test_ab_bc_never_implies_ac(self):
        skills = [artifact("a"), artifact("b"), artifact("c")]
        evidence = [judgment("a", "b", "duplicate"), judgment("b", "c", "duplicate")]
        graph = build_graph(skills, evidence)
        self.assertIsNone(graph.edge("a", "c"))
        plans = build_plans(skills, evidence)
        self.assert_star_shaped(graph, plans)
        by_canonical = {plan.canonical: plan for plan in plans}
        self.assertEqual(by_canonical["a"].absorbed, ("b",))
        self.assertNotIn("c", by_canonical["a"].absorbed)
        self.assertEqual(by_canonical["b"].absorbed, ("c",))
        self.assertFalse(by_canonical["b"].applicable)
        self.assertIn("canonical-absorbed-by:a", by_canonical["b"].blockers)

    def test_three_duplicates_assign_each_member_once(self):
        skills = [artifact("a"), artifact("b"), artifact("c")]
        evidence = [
            judgment("a", "b", "duplicate"),
            judgment("b", "c", "duplicate"),
            judgment("a", "c", "duplicate"),
        ]
        plans = build_plans(skills, evidence)
        self.assert_star_shaped(build_graph(skills, evidence), plans)
        self.assertEqual(len(plans), 1)
        self.assertEqual((plans[0].canonical, plans[0].absorbed), ("a", ("b", "c")))

    def test_stale_hash_is_refused(self):
        skills = [artifact("a"), artifact("b")]
        evidence = [judgment("a", "b", "duplicate", digest_a="old-digest")]
        edge = require_edge(build_graph(skills, evidence), "a", "b")
        self.assertFalse(edge.authorized)
        self.assertEqual(edge.refusals, ("stale-hash",))
        plans = build_plans(skills, evidence)
        self.assertEqual([plan.status for plan in plans], ["noop"])
        self.assertIn("stale-hash", plans[0].metadata["refusals"])

    def test_truncated_pair_is_refused(self):
        skills = [artifact("a"), artifact("b")]
        evidence = [judgment("a", "b", "duplicate")]
        graph = build_graph(skills, evidence, truncated_pairs=[pair_key("b", "a")])
        self.assertEqual(require_edge(graph, "a", "b").refusals, ("truncated",))
        self.assertFalse(require_edge(graph, "a", "b").authorized)
        plans = build_plans(skills, evidence, truncated_pairs=[pair_key("a", "b")])
        self.assertEqual([plan.status for plan in plans], ["noop"])

    def test_insufficient_evidence_never_authorizes(self):
        # the engine turns a truncated pair state into exactly this judgment
        skills = [artifact("a"), artifact("b")]
        evidence = [judgment("a", "b", "insufficient_evidence", confidence=1.0, coverage=0.0)]
        edge = require_edge(build_graph(skills, evidence), "a", "b")
        self.assertFalse(edge.authorized)
        self.assertIn("insufficient-evidence", edge.refusals)
        self.assertEqual(build_plans(skills, evidence)[0].status, "noop")

    def test_low_confidence_coverage_and_conflict_score_are_refused(self):
        skills = [artifact("a"), artifact("b")]
        cases: tuple[tuple[dict[str, Any], str], ...] = (
            ({"confidence": MIN_CONFIDENCE - 0.01}, "low-confidence"),
            ({"coverage": MIN_COVERAGE - 0.01}, "low-coverage"),
            ({"conflict": MAX_CONFLICT + 0.01}, "conflict-score"),
        )
        for fields, reason in cases:
            edge = require_edge(build_graph(skills, [judgment("a", "b", "duplicate", **fields)]), "a", "b")
            self.assertFalse(edge.authorized, reason)
            self.assertIn(reason, edge.refusals)
            self.assertEqual(build_plans(skills, [judgment("a", "b", "duplicate", **fields)])[0].status, "noop")

    def test_preservation_gate_blocks_absorption(self):
        skills = [artifact("a"), artifact("b")]
        subset = require_edge(build_graph(skills, [judgment("a", "b", "a_subset_of_b", preservation_a_in_b=0.4)]), "a", "b")
        self.assertEqual(subset.refusals, ("low-preservation",))
        duplicate = require_edge(build_graph(skills, [judgment("a", "b", "duplicate", preservation_b_in_a=0.4)]), "a", "b")
        self.assertEqual(duplicate.refusals, ("low-preservation",))

    def test_conflict_relation_and_conflict_inside_group_block(self):
        skills = [artifact("a"), artifact("b"), artifact("c")]
        conflicting = [judgment("a", "b", "conflict")]
        edge = require_edge(build_graph(skills, conflicting), "a", "b")
        self.assertEqual(edge.refusals, ("conflict",))
        self.assertTrue(edge.conflicted)
        self.assertEqual(build_plans(skills, conflicting)[0].status, "noop")

        group = [
            judgment("a", "b", "b_subset_of_a"),
            judgment("a", "c", "b_subset_of_a"),
            judgment("b", "c", "conflict"),
        ]
        plans = build_plans(skills, group)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].canonical, "a")
        self.assertEqual(plans[0].absorbed, ("b", "c"))
        self.assertFalse(plans[0].applicable)
        self.assertIn("conflict:b::c", plans[0].blockers)

    def test_protected_member_blocks_the_plan(self):
        skills = [artifact("a"), artifact("b", protected=("pinned",))]
        plans = build_plans(skills, [judgment("a", "b", "duplicate")])
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].status, "blocked")
        self.assertFalse(plans[0].applicable)
        self.assertIn("protected:b:pinned", plans[0].blockers)
        self.assertEqual(plans[0].absorbed, ("b",))

        pinned = [artifact("a", protected=("pinned", "cron-referenced")), artifact("b")]
        blocked = build_plans(pinned, [judgment("a", "b", "duplicate")])
        self.assertIn("protected:a:pinned,cron-referenced", blocked[0].blockers)

    def test_unknown_artifact_and_self_pair_are_refused(self):
        skills = [artifact("a")]
        ghost = require_edge(build_graph(skills, [judgment("a", "ghost", "duplicate")]), "a", "ghost")
        self.assertEqual(ghost.refusals, ("unknown-artifact",))
        self.assertFalse(ghost.authorized)
        itself = require_edge(build_graph(skills, [judgment("a", "a", "duplicate")]), "a", "a")
        self.assertEqual(itself.refusals, ("self-pair",))
        self.assertFalse(itself.authorized)
        self.assertEqual(build_plans(skills, [judgment("a", "a", "duplicate")])[0].status, "noop")

    def test_malformed_evidence_is_refused(self):
        skills = [artifact("a"), artifact("b")]
        evidence = [judgment("a", "b", "duplicate", coverage=1.4)]
        edge = require_edge(build_graph(skills, evidence), "a", "b")
        self.assertFalse(edge.authorized)
        self.assertIn("malformed-evidence", edge.refusals)

    def test_budget_cap_prefers_plans_that_absorb_more(self):
        skills = [artifact(name) for name in ("a", "b", "c", "d", "e")]
        evidence = [
            judgment("a", "b", "b_subset_of_a"),
            judgment("a", "c", "b_subset_of_a"),
            judgment("d", "e", "duplicate"),
        ]
        plans = build_plans(skills, evidence, max_plans=1)
        self.assertEqual(len(plans), 1)
        self.assertEqual((plans[0].canonical, plans[0].absorbed), ("a", ("b", "c")))
        self.assertEqual(plans, build_plans(list(reversed(skills)), list(reversed(evidence)), max_plans=1))

    def test_judgment_order_and_duplicate_keys_do_not_change_plans(self):
        skills = [artifact("a"), artifact("b"), artifact("c")]
        evidence = [
            judgment("a", "b", "a_subset_of_b", confidence=0.9),   # a into b
            judgment("a", "b", "b_subset_of_a", confidence=0.99),  # same key, opposite claim
            judgment("a", "c", "b_subset_of_a"),
        ]
        first = build_plans(skills, evidence)
        self.assertEqual(first, build_plans(list(reversed(skills)), list(reversed(evidence))))
        graph = build_graph(skills, evidence)
        edge = require_edge(graph, "a", "b")
        self.assertEqual(edge.confidence, 0.99)
        self.assertEqual((edge.canonical, edge.absorbed), ("a", "b"))
        self.assertEqual([(plan.canonical, plan.absorbed) for plan in first], [("a", ("b", "c"))])
        self.assert_star_shaped(graph, first)

    def test_noop_plan_is_explicit_and_stable(self):
        skills = [artifact("a"), artifact("b")]
        first = build_plans(skills, [])
        self.assertEqual(len(first), 1)
        plan = first[0]
        self.assertEqual(plan.status, "noop")
        self.assertEqual((plan.canonical, plan.canonical_digest), ("", ""))
        self.assertEqual((plan.absorbed, plan.relation_keys), ((), ()))
        self.assertFalse(plan.applicable)
        self.assertEqual(plan.metadata["reason"], "no-authorized-relations")
        self.assertEqual(first, build_plans(list(reversed(skills)), []))


if __name__ == "__main__":
    unittest.main()
