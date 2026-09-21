from __future__ import annotations

import unittest
from pathlib import Path

from plugin.candidates import deterministic_relation, generate_candidates
from plugin.models import SkillArtifact


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


if __name__ == "__main__":
    unittest.main()
