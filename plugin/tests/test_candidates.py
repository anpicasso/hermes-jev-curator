from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plugin.candidates import deterministic_relation, generate_candidates
from plugin.models import SkillArtifact
from plugin.questions import has_truncation, pair_state


def artifact(name: str, text: str, digest: str) -> SkillArtifact:
    return SkillArtifact(name=name, path=Path("/skills") / name, description=text[:40], text=text, digest=digest)


class CandidateTests(unittest.TestCase):
    def test_deterministic_top_k_and_pair_dedup(self):
        skills = [
            artifact("git-review", "Review git diffs and inspect branch history safely", "a"),
            artifact("git-pr", "Review pull request git diffs and branch history safely", "b"),
            artifact("weather", "Fetch current weather forecasts", "c"),
        ]
        first = generate_candidates(skills, top_k=1, max_pairs=10)
        second = generate_candidates(reversed(skills), top_k=1, max_pairs=10)
        self.assertEqual(first, second)
        self.assertEqual((first[0].a, first[0].b), ("git-pr", "git-review"))
        self.assertIn(deterministic_relation(first[0]), {"duplicate", "same_class", "unrelated"})

    def test_explicit_head_tail_truncation(self):
        a = artifact("a", "a" * 10_000, "a")
        b = artifact("b", "b" * 10_000, "b")
        state = pair_state(a, b, 4_000)
        self.assertTrue(has_truncation(state))
        self.assertIn("explicitly truncated", state["skill_a"])
        self.assertTrue(state["skill_a"].startswith("a"))
        self.assertTrue(state["skill_a"].endswith("a"))

    def test_outbound_pair_state_redacts_secrets_and_omits_package_digests(self):
        secret = "sk-" + "A" * 20
        a = artifact("a", f"run with api_key={secret}", "private-digest-a")
        b = artifact("b", "ordinary procedure", "private-digest-b")
        state = pair_state(a, b, 4_000)

        blob = repr(state)
        self.assertNotIn(secret, blob)
        self.assertNotIn("private-digest", blob)
        self.assertNotIn("skill_a_digest", state)
        self.assertNotIn(secret, state["skill_a"])
        self.assertNotEqual(state["skill_a"], a.text)


if __name__ == "__main__":
    unittest.main()
