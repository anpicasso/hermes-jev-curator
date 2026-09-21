"""Offline tests for the frozen semantic-contract corpus and its evaluator."""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.evaluate import (  # noqa: E402
    DIRECT,
    RELATIONS,
    corpus_thresholds,
    evaluate,
    expected_answers,
    expected_responses,
    load_corpus,
    load_responses,
    main,
    merge_decision,
    pair_state,
    print_report,
    self_check,
)

CORPUS_PATH = ROOT / "benchmarks" / "corpus.json"


def _corpus() -> dict:
    return load_corpus(CORPUS_PATH)


def _row(report: dict, case_id: str) -> dict:
    return next(row for row in report["rows"] if row["id"] == case_id)


def _confident_duplicate() -> dict:
    """A maximally merge-friendly response, used to prove the gate still refuses bad cases."""
    probabilities = {relation: 0.0 for relation in RELATIONS}
    probabilities["duplicate"] = 1.0
    return {
        "relation": {"choice": "duplicate", "confidence": 0.99, "probabilities": probabilities},
        "coverage": {"noul": 0.99},
        "a_in_b": {"noul": 0.99},
        "b_in_a": {"noul": 0.99},
        "conflict": {"noul": 0.0},
        "same_class": {"noul": 0.0},
    }


class CorpusTests(unittest.TestCase):
    def test_corpus_covers_every_contract_relation(self):
        corpus = _corpus()
        labels = {pair["relation"] for pair in corpus["pairs"]}
        self.assertEqual(labels, set(RELATIONS))
        for relation in RELATIONS:
            self.assertGreaterEqual(sum(1 for p in corpus["pairs"] if p["relation"] == relation), 2,
                                    f"relation {relation} needs at least two cases")
        self.assertTrue(any(p.get("adversarial") for p in corpus["pairs"]))
        self.assertEqual(corpus["contract_version"], "skill-relations-v1")

    def test_merge_authorization_is_only_claimed_for_direct_containment(self):
        for pair in _corpus()["pairs"]:
            if pair["merge_authorized"]:
                self.assertIn(pair["relation"], DIRECT, pair["id"])
                self.assertFalse(pair.get("truncated", False), pair["id"])
                self.assertFalse(pair.get("adversarial", False), pair["id"])

    def test_corpus_is_synthetic_and_carries_no_credentials(self):
        raw = CORPUS_PATH.read_text(encoding="utf-8")
        forbidden = (
            r"sk-[A-Za-z0-9]{12,}",
            r"ghp_[A-Za-z0-9]{12,}",
            r"Bearer\s+[A-Za-z0-9._-]{8,}",
            r"://[^/\s]+:[^/\s]+@",
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            r"\b(?:api[_-]?key|access[_-]?token|password)\b\s*[:=]\s*\S+",
        )
        for pattern in forbidden:
            self.assertIsNone(re.search(pattern, raw, re.I), f"corpus matches {pattern!r}")
        for pair in _corpus()["pairs"]:
            self.assertTrue(pair["id"] and pair["callout"], pair["id"])

    def test_pair_state_flags_truncation_explicitly(self):
        corpus = _corpus()
        truncated = [p for p in corpus["pairs"] if p.get("truncated")]
        self.assertTrue(truncated)
        for pair in truncated:
            state = pair_state(pair)
            self.assertIn("true", (state["skill_a_truncated"], state["skill_b_truncated"]))
            self.assertIn("explicitly truncated", state["skill_a"] + state["skill_b"])
        state = pair_state(next(p for p in corpus["pairs"] if not p.get("truncated")))
        self.assertEqual((state["skill_a_truncated"], state["skill_b_truncated"]), ("false", "false"))
        self.assertEqual(state["contract"], "skill-relations-v1")


class EvaluatorTests(unittest.TestCase):
    def test_embedded_expected_answers_score_perfectly(self):
        corpus = _corpus()
        report = evaluate(corpus, expected_responses(corpus))
        self.assertEqual(report["n_cases"], len(corpus["pairs"]))
        self.assertEqual(report["label_accuracy"], 1.0)
        self.assertEqual(report["false_merge_authorizations"], 0)
        self.assertEqual(report["missed_merges"], 0)
        self.assertTrue(report["adversarial_ok"])

    def test_self_check_passes_and_emits_embedded_responses(self):
        corpus = _corpus()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "expected.json"
            with contextlib.redirect_stdout(io.StringIO()):
                report = self_check(corpus, emit=target)
            self.assertEqual(report["label_accuracy"], 1.0)
            emitted = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(set(emitted), {p["id"] for p in corpus["pairs"]})
            self.assertEqual(emitted, expected_responses(corpus))

    def test_merge_gate_refuses_every_degraded_variant(self):
        thresholds = corpus_thresholds(_corpus())
        clean = merge_decision("duplicate", 0.95, 0.95, 0.95, 0.95, 0.05,
                               truncated=False, thresholds=thresholds)
        self.assertEqual(clean, (True, "authorized"))
        refusals = {
            "low-confidence": merge_decision("duplicate", 0.5, 0.95, 0.95, 0.95, 0.05,
                                             truncated=False, thresholds=thresholds),
            "low-coverage": merge_decision("duplicate", 0.95, 0.5, 0.95, 0.95, 0.05,
                                           truncated=False, thresholds=thresholds),
            "high-conflict": merge_decision("duplicate", 0.95, 0.95, 0.95, 0.95, 0.9,
                                            truncated=False, thresholds=thresholds),
            "one-way-preservation": merge_decision("duplicate", 0.95, 0.95, 0.95, 0.1, 0.05,
                                                   truncated=False, thresholds=thresholds),
            "same-class": merge_decision("same_class", 0.99, 0.99, 0.99, 0.99, 0.0,
                                         truncated=False, thresholds=thresholds),
            "complementary": merge_decision("complementary", 0.99, 0.99, 0.99, 0.99, 0.0,
                                            truncated=False, thresholds=thresholds),
            "conflict": merge_decision("conflict", 0.99, 0.99, 0.99, 0.99, 0.99,
                                       truncated=False, thresholds=thresholds),
            "unrelated": merge_decision("unrelated", 0.99, 0.99, 0.99, 0.99, 0.0,
                                        truncated=False, thresholds=thresholds),
            "insufficient-evidence": merge_decision("insufficient_evidence", 0.99, 0.99, 0.99, 0.99, 0.0,
                                                    truncated=False, thresholds=thresholds),
            "truncated": merge_decision("duplicate", 0.99, 0.99, 0.99, 0.99, 0.0,
                                        truncated=True, thresholds=thresholds),
            "missing-answers": merge_decision(None, None, None, None, None, None,
                                              truncated=False, thresholds=thresholds),
        }
        for label, (authorized, reason) in refusals.items():
            self.assertFalse(authorized, f"{label} was authorized")
            self.assertNotEqual(reason, "authorized", label)

    def test_directional_containment_requires_the_matching_side(self):
        thresholds = corpus_thresholds(_corpus())
        authorized, _ = merge_decision("a_subset_of_b", 0.95, 0.95, 0.95, 0.05, 0.05,
                                       truncated=False, thresholds=thresholds)
        self.assertTrue(authorized)
        authorized, reason = merge_decision("a_subset_of_b", 0.95, 0.95, 0.05, 0.95, 0.05,
                                            truncated=False, thresholds=thresholds)
        self.assertEqual((authorized, reason), (False, "preservation-a"))
        authorized, reason = merge_decision("b_subset_of_a", 0.95, 0.95, 0.05, 0.95, 0.05,
                                            truncated=False, thresholds=thresholds)
        self.assertTrue(authorized)
        authorized, reason = merge_decision("b_subset_of_a", 0.95, 0.95, 0.95, 0.05, 0.05,
                                            truncated=False, thresholds=thresholds)
        self.assertEqual((authorized, reason), (False, "preservation-b"))

    def test_adversarial_lossy_merges_are_never_authorized(self):
        corpus = _corpus()
        adversarial = [p for p in corpus["pairs"] if p.get("adversarial")]
        self.assertTrue(adversarial)
        for pair in adversarial:
            # A confident-but-wrong "duplicate" verdict must never pass silently: the gate
            # refuses truncated cases outright and the evaluator flags the rest as false merges.
            report = evaluate(corpus, {pair["id"]: _confident_duplicate()})
            row = _row(report, pair["id"])
            self.assertFalse(row["expected_merge"], pair["id"])
            self.assertFalse(row["relation_ok"], pair["id"])
            if pair.get("truncated"):
                self.assertFalse(row["merge_authorized"], pair["id"])
                self.assertFalse(row["false_merge"], pair["id"])
            else:
                self.assertTrue(row["false_merge"], pair["id"])
                self.assertEqual(report["false_merge_ids"], [pair["id"]])
            self.assertFalse(report["adversarial_ok"], pair["id"])
        # A *wrong* confident verdict is likewise caught as a false merge.
        same_class_pair = next(p for p in corpus["pairs"]
                               if p["relation"] == "same_class" and not p.get("truncated"))
        row = _row(evaluate(corpus, {same_class_pair["id"]: _confident_duplicate()}), same_class_pair["id"])
        self.assertTrue(row["merge_authorized"])
        self.assertTrue(row["false_merge"])
        self.assertFalse(row["relation_ok"])

    def test_false_merge_is_flagged_and_exit_code_is_nonzero(self):
        corpus = _corpus()
        pair = next(p for p in corpus["pairs"] if p["relation"] == "conflict")
        with tempfile.TemporaryDirectory() as tmp:
            responses = Path(tmp) / "responses.json"
            responses.write_text(json.dumps({pair["id"]: _confident_duplicate()}), encoding="utf-8")
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--responses", str(responses)])
            self.assertEqual(code, 1)
            self.assertIn("FALSE MERGES", buffer.getvalue())
            report = evaluate(corpus, {pair["id"]: _confident_duplicate()})
            self.assertEqual(report["false_merge_authorizations"], 1)
            self.assertEqual(report["false_merge_ids"], [pair["id"]])

    def test_cli_self_check_and_perfect_responses_exit_zero(self):
        corpus = _corpus()
        with tempfile.TemporaryDirectory() as tmp:
            responses = Path(tmp) / "expected.json"
            responses.write_text(json.dumps(expected_responses(corpus)), encoding="utf-8")
            for argv in (["--self-check"], ["--responses", str(responses)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(argv), 0, argv)

    def test_response_shapes_are_all_loadable(self):
        corpus = _corpus()
        expected = expected_responses(corpus)
        case_id = corpus["pairs"][0]["id"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mapping.json").write_text(json.dumps(expected), encoding="utf-8")
            (root / "nested.json").write_text(
                json.dumps({key: {"answers": value} for key, value in expected.items()}), encoding="utf-8")
            loaded, notes = load_responses(root / "mapping.json", corpus)
            self.assertEqual(len(loaded), len(corpus["pairs"]))
            self.assertEqual(loaded, {key: value for key, value in expected.items()})
            loaded, _ = load_responses(root / "nested.json", corpus)
            self.assertEqual(loaded, {key: value for key, value in expected.items()})
            directory = root / "per_case"
            directory.mkdir()
            for pair in corpus["pairs"]:
                (directory / f"{pair['id']}.json").write_text(
                    json.dumps({"answers": expected[pair["id"]]}), encoding="utf-8")
            loaded, notes = load_responses(directory, corpus)
            self.assertEqual(len(loaded), len(corpus["pairs"]))
            self.assertEqual(notes, [])

            # A single full response matched by digests, plus a truncated case left unmatched.
            pair = corpus["pairs"][0]
            digests = {side: f"sha256:{pair['id']}-{side}" for side in ("a", "b")}
            pair_with_digests = {**pair, "a": {**pair["a"], "digest": digests["a"]},
                                 "b": {**pair["b"], "digest": digests["b"]}}
            corpus_digests = {**corpus, "pairs": [pair_with_digests]}
            response = {"model": "jev-latest", "answers": {
                "state": {"skill_a_digest": digests["a"], "skill_b_digest": digests["b"]},
                **expected[case_id]}}
            single = root / "single.json"
            single.write_text(json.dumps(response), encoding="utf-8")
            loaded, notes = load_responses(single, corpus_digests)
            self.assertEqual(list(loaded), [case_id])
            self.assertEqual(loaded[case_id]["relation"], expected[case_id]["relation"])

            (root / "empty.json").write_text(json.dumps({}), encoding="utf-8")
            loaded, notes = load_responses(root / "empty.json", corpus)
            self.assertEqual(loaded, {})
            self.assertEqual(len(notes), len(corpus["pairs"]))

    def test_missing_and_invalid_answers_never_authorize(self):
        corpus = _corpus()
        report = evaluate(corpus, {})
        self.assertEqual(report["coverage"], 0.0)
        self.assertEqual(report["false_merge_authorizations"], 0)
        self.assertEqual(report["missed_merges"], sum(1 for p in corpus["pairs"] if p["merge_authorized"]))
        broken = {"relation": {"choice": "made-up", "confidence": 0.99},
                  "coverage": {"noul": 0.99}, "a_in_b": {"noul": 0.99},
                  "b_in_a": {"noul": 0.99}, "conflict": {"noul": 0.0}}
        report = evaluate(corpus, {p["id"]: broken for p in corpus["pairs"]})
        self.assertEqual(report["label_accuracy"], 0.0)
        self.assertEqual(report["false_merge_authorizations"], 0)

        case_id = corpus["pairs"][0]["id"]
        for invalid in (float("nan"), float("inf"), -0.1, 1.1):
            malformed = _confident_duplicate()
            malformed["coverage"]["noul"] = invalid
            row = _row(evaluate(corpus, {case_id: malformed}), case_id)
            self.assertFalse(row["merge_authorized"], invalid)

        bad_probabilities = expected_answers(corpus["pairs"][0])
        bad_probabilities["relation"]["probabilities"] = {"duplicate": 1.0}
        row = _row(evaluate(corpus, {case_id: bad_probabilities}), case_id)
        self.assertFalse(row["merge_authorized"])

    def test_report_and_text_output_are_deterministic(self):
        corpus = _corpus()
        responses = expected_responses(corpus)
        first = evaluate(corpus, responses)
        second = evaluate(corpus, responses)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        buffer_a, buffer_b = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buffer_a):
            print_report(first)
        with contextlib.redirect_stdout(buffer_b):
            print_report(second)
        self.assertEqual(buffer_a.getvalue(), buffer_b.getvalue())
        n = first["n_cases"]
        self.assertIn(f"label accuracy {n}/{n} = 1.000", buffer_a.getvalue())
        self.assertIn("false merge-authorizations 0", buffer_a.getvalue())
        self.assertIn("all refused and labeled: True", buffer_a.getvalue())

    def test_expected_answers_respect_the_contract_shape(self):
        corpus = _corpus()
        for pair in corpus["pairs"]:
            answers = expected_answers(pair)
            self.assertEqual(set(answers), {"relation", "coverage", "a_in_b", "b_in_a", "conflict", "same_class"})
            self.assertEqual(answers["relation"]["choice"], pair["relation"])
            self.assertEqual(set(answers["relation"]["probabilities"]), set(RELATIONS))
            self.assertAlmostEqual(sum(answers["relation"]["probabilities"].values()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
