#!/usr/bin/env python3
"""Frozen offline evaluator for the skill-relations-v1 semantic contract corpus.

Reads `benchmarks/corpus.json` (synthetic, non-identifying skill pairs) plus
either saved Jev response JSON or a plain response mapping, then reports
deterministic metrics: exact-label accuracy and false merge-authorizations.

Merge authorization is conservative by design and mirrors the plugin's
guardrails: coverage high, confidence high, direct containment/duplicate,
low conflict, no truncation. Anything less is refused.

Stdlib only. No network. Read-only apart from optional `--emit`/`--self-check`
output paths.

Usage:
  python3 benchmarks/evaluate.py --self-check
  python3 benchmarks/evaluate.py --responses responses.json
  python3 benchmarks/evaluate.py --responses responses/ --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "corpus.json"

# Reuse the shipped contract and policy. A benchmark-only approximation can
# otherwise authorize evidence that the plugin itself rejects.
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from plugin.graph import (  # noqa: E402
    MAX_CONFLICT,
    MIN_CONFIDENCE,
    MIN_COVERAGE,
    MIN_PRESERVATION,
)
from plugin.questions import relation_questions  # noqa: E402
from plugin.transport import validate_answers  # noqa: E402

# Kept explicit so corpus coverage remains obvious; tests assert parity with the
# production question criteria.
RELATIONS = (
    "duplicate",
    "a_subset_of_b",
    "b_subset_of_a",
    "same_class",
    "complementary",
    "conflict",
    "unrelated",
    "insufficient_evidence",
)
DIRECT = frozenset({"duplicate", "a_subset_of_b", "b_subset_of_a"})
NUOL_KEYS = ("coverage", "a_in_b", "b_in_a", "conflict", "same_class")
DEFAULT_THRESHOLDS = {
    "confidence_min": MIN_CONFIDENCE,
    "coverage_min": MIN_COVERAGE,
    "preservation_min": MIN_PRESERVATION,
    "conflict_max": MAX_CONFLICT,
}
# Sentinel digest returned when a response does not name the digests it judged.
_NO_DIGEST = ""


def load_corpus(path: Path) -> dict[str, Any]:
    corpus = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(corpus, dict) or not isinstance(corpus.get("pairs"), list) or not corpus["pairs"]:
        raise ValueError("corpus must be an object with a non-empty 'pairs' list")
    seen: set[str] = set()
    for pair in corpus["pairs"]:
        missing = [key for key in ("id", "relation", "merge_authorized", "a", "b") if key not in pair]
        if missing:
            raise ValueError(f"pair is missing {missing}")
        if pair["id"] in seen:
            raise ValueError(f"duplicate pair id {pair['id']!r}")
        seen.add(pair["id"])
        if pair["relation"] not in RELATIONS:
            raise ValueError(f"pair {pair['id']!r} has unknown relation {pair['relation']!r}")
        for side in ("a", "b"):
            if not isinstance(pair[side], dict) or not pair[side].get("text"):
                raise ValueError(f"pair {pair['id']!r} side {side!r} needs a text body")
    return corpus


def corpus_thresholds(corpus: Mapping[str, Any]) -> dict[str, float]:
    raw = corpus.get("thresholds") or {}
    thresholds = dict(DEFAULT_THRESHOLDS)
    for key in thresholds:
        if key in raw:
            thresholds[key] = float(raw[key])
    return thresholds


def pair_state(pair: Mapping[str, Any]) -> dict[str, str]:
    """Mirror of plugin.questions.pair_state: named, bounded, truncation explicit."""
    a, b = pair["a"], pair["b"]
    return {
        "contract": "skill-relations-v1",
        "skill_a_name": str(a["name"]),
        "skill_a_digest": str(a.get("digest", _NO_DIGEST)),
        "skill_a": str(a["text"]),
        "skill_a_truncated": str(bool(a.get("truncated", pair.get("truncated", False)))).lower(),
        "skill_b_name": str(b["name"]),
        "skill_b_digest": str(b.get("digest", _NO_DIGEST)),
        "skill_b": str(b["text"]),
        "skill_b_truncated": str(bool(b.get("truncated", pair.get("truncated", False)))).lower(),
    }


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _validated_answers(value: Any) -> Mapping[str, Any] | None:
    """Use the production typed-response validator; malformed evidence is absent."""
    if not isinstance(value, Mapping):
        return None
    try:
        return validate_answers(value, relation_questions())
    except (RuntimeError, TypeError, ValueError):
        return None


def _answer_value(answer: Any, kind: str) -> float | None:
    if not isinstance(answer, Mapping):
        return None
    if kind == "choice":
        return None
    return _as_float(answer.get("noul", answer.get("confidence")))


def _choice(answer: Any) -> str | None:
    if not isinstance(answer, Mapping):
        return None
    choice = answer.get("choice")
    return choice if isinstance(choice, str) else None


def _norm_digest(value: Any) -> str:
    return value if isinstance(value, str) else _NO_DIGEST


def merge_decision(
    relation: str | None,
    confidence: float | None,
    coverage: float | None,
    a_in_b: float | None,
    b_in_a: float | None,
    conflict: float | None,
    *,
    truncated: bool,
    thresholds: Mapping[str, float],
) -> tuple[bool, str]:
    """Conservative merge gate. Returns (authorized, reason)."""
    if truncated:
        return False, "truncated"
    if relation not in DIRECT:
        return False, "relation-not-direct"
    if confidence is None or confidence < thresholds["confidence_min"]:
        return False, "low-confidence"
    if coverage is None or coverage < thresholds["coverage_min"]:
        return False, "low-coverage"
    if conflict is None or conflict > thresholds["conflict_max"]:
        return False, "conflict"
    if relation in {"a_subset_of_b", "duplicate"} and (a_in_b is None or a_in_b < thresholds["preservation_min"]):
        return False, "preservation-a"
    if relation in {"b_subset_of_a", "duplicate"} and (b_in_a is None or b_in_a < thresholds["preservation_min"]):
        return False, "preservation-b"
    return True, "authorized"


def _first_answer(entry: Mapping[str, Any], case_id: str) -> Mapping[str, Any] | None:
    """Accept one case's answer object, {case_id: answers}, or a full Jev response."""
    answers = entry.get("answers")
    if isinstance(answers, Mapping):
        return answers  # full Jev response payload for one case
    if case_id in entry and isinstance(entry[case_id], Mapping):
        return _first_answer(entry[case_id], case_id) or entry[case_id]
    if _is_answers(entry):
        return entry  # bare answers mapping
    return None


def _load_answers_from(obj: Any, case_id: str) -> Mapping[str, Any] | None:
    if not isinstance(obj, Mapping):
        return None
    found = _first_answer(obj, case_id)
    if found is not None:
        return found
    responses = obj.get("responses")
    if isinstance(responses, Mapping):
        inner = responses.get(case_id)
        if isinstance(inner, Mapping):
            return _first_answer(inner, case_id) or inner
    return None


def _is_answers(obj: Mapping[str, Any]) -> bool:
    return "relation" in obj or "coverage" in obj


def load_responses(path: Path, corpus: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """Return {case_id: answers} plus a list of notes about what was (not) loaded.

    Accepted shapes:
      * directory of *.json files, each keyed by case id, file stem, or holding a full response
      * file mapping case_id -> answers (or -> {"answers": ...})
      * file holding one full Jev response; mapped to its matching pair by digest/name,
        or to the single pair when the corpus has exactly one.
    """
    path = Path(path)
    loaded: dict[str, Mapping[str, Any]] = {}
    notes: list[str] = []
    files: list[Path] = sorted(path.glob("*.json")) if path.is_dir() else [path]
    case_ids = {pair["id"] for pair in corpus["pairs"]}

    for file in files:
        try:
            obj = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            notes.append(f"skipped {file.name}: {exc}")
            continue
        if not isinstance(obj, Mapping):
            notes.append(f"skipped {file.name}: not a JSON object")
            continue

        # A full Jev response carries a top-level "answers" object: match it by identity.
        if isinstance(obj.get("answers"), Mapping):
            matched = {pair["id"]: _match_by_identity(obj, pair) for pair in corpus["pairs"]}
            matched = {case_id: answers for case_id, answers in matched.items() if answers is not None}
            if matched:
                loaded.update(matched)
                continue
            if len(corpus["pairs"]) == 1:
                loaded[corpus["pairs"][0]["id"]] = obj["answers"]
                continue
            if file.stem in case_ids:
                loaded[file.stem] = obj["answers"]  # per-case file named by case id
                continue
            notes.append(f"{file.name}: response matched no corpus pair by digest or skill name")
            continue

        # A mapping keyed by case id, a bare answers object for the file's own case, or a wrapper.
        for pair in corpus["pairs"]:
            case_id = pair["id"]
            answers = _load_answers_from(obj, case_id)
            if answers is None and file.stem == case_id and _is_answers(obj):
                answers = obj
            if answers is not None:
                loaded[case_id] = answers

    for pair in corpus["pairs"]:
        if pair["id"] not in loaded:
            notes.append(f"no response found for case {pair['id']!r}")
    return loaded, notes


def _match_by_identity(obj: Mapping[str, Any], pair: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Match a single full response to a pair by digests, else by skill names."""
    answers = obj.get("answers")
    if not isinstance(answers, Mapping):
        return None
    a, b = pair["a"], pair["b"]
    digest_a, digest_b = str(a.get("digest", _NO_DIGEST)), str(b.get("digest", _NO_DIGEST))
    if digest_a and digest_b:
        state = answers.get("state")
        if isinstance(state, Mapping):
            if _norm_digest(state.get("skill_a_digest")) == digest_a and _norm_digest(state.get("skill_b_digest")) == digest_b:
                return answers
            return None
    names = answers.get("skill_names")
    if isinstance(names, Mapping):
        if str(names.get("a", "")) == str(a["name"]) and str(names.get("b", "")) == str(b["name"]):
            return answers
    return None


def expected_answers(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Embedded expected answers used by --self-check and by corpus-driven checks."""
    relation = pair["relation"]
    authorized = bool(pair["merge_authorized"])
    direct = relation in DIRECT and not pair.get("truncated", False)
    strong = 0.97 if authorized else 0.62
    mid = 0.95 if authorized else 0.58
    weak = 0.04 if authorized else 0.88
    if pair.get("truncated", False):
        confidence = 0.35
        coverage = 0.22
    elif relation == "conflict":
        confidence, coverage = 0.93, 0.9
    elif relation in {"same_class", "complementary"}:
        confidence, coverage = 0.9, 0.85
    elif relation == "unrelated":
        confidence, coverage = 0.97, 0.96
    else:
        confidence, coverage = strong, mid
    if pair.get("adversarial") and not authorized:
        confidence = min(confidence, 0.74)
    a_in_b = mid if (authorized or relation == "a_subset_of_b") else weak
    b_in_a = mid if (authorized or relation == "b_subset_of_a") else weak
    conflict = 0.9 if relation == "conflict" else (0.05 if authorized else 0.1)
    probabilities = {option: round((1.0 - confidence) / (len(RELATIONS) - 1), 6) for option in RELATIONS}
    probabilities[relation] = confidence
    return {
        "relation": {"choice": relation, "confidence": confidence, "probabilities": probabilities},
        "coverage": {"noul": coverage},
        "a_in_b": {"noul": a_in_b},
        "b_in_a": {"noul": b_in_a},
        "conflict": {"noul": conflict},
        "same_class": {"noul": 0.9 if relation == "same_class" else 0.1},
    }


def expected_responses(corpus: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {pair["id"]: expected_answers(pair) for pair in corpus["pairs"]}


def evaluate(corpus: Mapping[str, Any], responses: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = corpus_thresholds(corpus)
    pairs = corpus["pairs"]
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        case_id = pair["id"]
        answers = _validated_answers(responses.get(case_id))
        state = pair_state(pair)
        truncated = "true" in (state["skill_a_truncated"], state["skill_b_truncated"])
        relation_answer = answers.get("relation") if isinstance(answers, Mapping) else None
        got_choice = _choice(relation_answer)
        confidence = _as_float(relation_answer.get("confidence")) if isinstance(relation_answer, Mapping) else None
        nouls = {key: _answer_value(answers.get(key) if isinstance(answers, Mapping) else None, "noul") for key in NUOL_KEYS}
        authorized, reason = merge_decision(
            got_choice, confidence, nouls["coverage"], nouls["a_in_b"], nouls["b_in_a"], nouls["conflict"],
            truncated=truncated, thresholds=thresholds,
        )
        rows.append({
            "id": case_id,
            "expected_relation": pair["relation"],
            "got_relation": got_choice,
            "relation_ok": got_choice == pair["relation"],
            "expected_merge": bool(pair["merge_authorized"]),
            "merge_authorized": authorized,
            "merge_reason": reason,
            "false_merge": authorized and not pair["merge_authorized"],
            "missed_merge": (not authorized) and bool(pair["merge_authorized"]),
            "confidence": confidence,
            "coverage": nouls["coverage"],
            "conflict": nouls["conflict"],
            "adversarial": bool(pair.get("adversarial", False)),
        })

    total = len(rows)
    answered = sum(1 for row in rows if row["got_relation"] is not None)
    correct = sum(1 for row in rows if row["relation_ok"])
    false_merges = [row["id"] for row in rows if row["false_merge"]]
    missed = [row["id"] for row in rows if row["missed_merge"]]
    adversarial = [row for row in rows if row["adversarial"]]
    per_relation: dict[str, dict[str, Any]] = {}
    for relation in RELATIONS:
        bucket = [row for row in rows if row["expected_relation"] == relation]
        if not bucket:
            continue
        per_relation[relation] = {
            "n": len(bucket),
            "correct": sum(1 for row in bucket if row["relation_ok"]),
            "accuracy": round(sum(1 for row in bucket if row["relation_ok"]) / len(bucket), 6),
            "false_merges": sum(1 for row in bucket if row["false_merge"]),
        }
    return {
        "corpus_version": corpus.get("corpus_version", "unknown"),
        "contract_version": corpus.get("contract_version", "unknown"),
        "thresholds": thresholds,
        "n_cases": total,
        "n_answered": answered,
        "n_correct": correct,
        "label_accuracy": round(correct / total, 6) if total else 0.0,
        "coverage": round(answered / total, 6) if total else 0.0,
        "expected_merges": sum(1 for row in rows if row["expected_merge"]),
        "authorized_merges": sum(1 for row in rows if row["merge_authorized"]),
        "false_merge_authorizations": len(false_merges),
        "missed_merges": len(missed),
        "false_merge_ids": sorted(false_merges),
        "missed_merge_ids": sorted(missed),
        "adversarial_n": len(adversarial),
        "adversarial_ok": all(row["relation_ok"] and not row["merge_authorized"] for row in adversarial),
        "per_relation": per_relation,
        "confusion": _confusion(rows),
        "rows": rows,
    }


def _confusion(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        counts.setdefault(row["expected_relation"], Counter())[str(row["got_relation"])] += 1
    return {expected: dict(sorted(got.items())) for expected, got in sorted(counts.items())}


def print_report(report: Mapping[str, Any]) -> None:
    print(f"corpus {report['corpus_version']} | contract {report['contract_version']}")
    print(f"cases {report['n_cases']} | answered {report['n_answered']} | coverage {report['coverage']:.3f}")
    print(f"label accuracy {report['n_correct']}/{report['n_cases']} = {report['label_accuracy']:.3f}")
    print(f"false merge-authorizations {report['false_merge_authorizations']}"
          f" | missed merges {report['missed_merges']}"
          f" | expected {report['expected_merges']} authorized {report['authorized_merges']}")
    if report["false_merge_ids"]:
        print(f"  FALSE MERGES: {', '.join(report['false_merge_ids'])}")
    if report["missed_merge_ids"]:
        print(f"  missed: {', '.join(report['missed_merge_ids'])}")
    print(f"adversarial {report['adversarial_n']} cases | all refused and labeled: {report['adversarial_ok']}")
    print("per-relation accuracy:")
    for relation, stats in report["per_relation"].items():
        print(f"  {relation:<22} {stats['correct']}/{stats['n']} = {stats['accuracy']:.3f}"
              f"  false-merges {stats['false_merges']}")
    print("per-case:")
    for row in report["rows"]:
        flag = "!" if (not row["relation_ok"] or row["false_merge"]) else " "
        got = row["got_relation"] or "<missing>"
        auth = "MERGE" if row["merge_authorized"] else "no-merge"
        print(f" {flag} {row['id']:<12} expected {row['expected_relation']:<22} got {got:<22} {auth} ({row['merge_reason']})")
    print("confusion (expected -> got):")
    for expected, got_counts in report["confusion"].items():
        rendered = ", ".join(f"{got}:{count}" for got, count in got_counts.items())
        print(f"  {expected:<22} {rendered}")


def self_check(corpus: Mapping[str, Any], *, emit: Path | None = None) -> dict[str, Any]:
    """Verify the corpus and evaluator against embedded expected answers."""
    problems: list[str] = []
    responses = expected_responses(corpus)
    if emit is not None:
        emit.write_text(json.dumps(responses, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = evaluate(corpus, responses)
    if report["n_cases"] != len(corpus["pairs"]):
        problems.append("case count mismatch")
    if report["label_accuracy"] != 1.0:
        problems.append(f"embedded answers scored {report['label_accuracy']:.3f} label accuracy")
    if report["false_merge_authorizations"] != 0:
        problems.append(f"false merge authorizations on embedded answers: {report['false_merge_ids']}")
    if report["missed_merges"] != 0:
        problems.append(f"embedded answers missed expected merges: {report['missed_merge_ids']}")
    if not report["adversarial_ok"]:
        problems.append("adversarial cases were not all refused")

    # Negative controls: the gate must refuse each degraded variant of a good case.
    good = next(pair for pair in corpus["pairs"] if pair["merge_authorized"])
    variants = {
        "low-confidence": lambda a: a["relation"].__setitem__("confidence", 0.3),
        "low-coverage": lambda a: a["coverage"].__setitem__("noul", 0.2),
        "high-conflict": lambda a: a["conflict"].__setitem__("noul", 0.95),
        "conflict-relation": lambda a: a["relation"].__setitem__("choice", "conflict"),
        "truncated": lambda a: None,  # handled by state, not answers
        "non-finite": lambda a: a["coverage"].__setitem__("noul", float("nan")),
        "bad-probabilities": lambda a: a["relation"]["probabilities"].clear(),
    }
    for label, mutate in variants.items():
        answers = json.loads(json.dumps(responses[good["id"]]))
        mutate(answers)
        if label == "truncated":
            got, _reason = merge_decision(
                answers["relation"]["choice"], answers["relation"]["confidence"],
                answers["coverage"]["noul"], answers["a_in_b"]["noul"], answers["b_in_a"]["noul"],
                answers["conflict"]["noul"], truncated=True, thresholds=corpus_thresholds(corpus),
            )
            if got:
                problems.append("truncation did not refuse a merge")
            continue
        report_one = evaluate({"pairs": [pair_copy(good)], **{k: v for k, v in corpus.items() if k != "pairs"}},
                              {good["id"]: answers})
        if report_one["rows"][0]["merge_authorized"]:
            problems.append(f"degraded variant {label!r} was still authorized")
    if problems:
        raise SystemExit("self-check FAILED:\n  " + "\n  ".join(problems))
    return report


def pair_copy(pair: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(pair))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="corpus JSON (default: benchmarks/corpus.json)")
    parser.add_argument("--responses", type=Path, help="saved Jev response JSON file or directory")
    parser.add_argument("--self-check", action="store_true", help="score embedded expected answers and verify the corpus")
    parser.add_argument("--emit", type=Path, help="with --self-check: write the embedded responses to this file")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    if args.self_check:
        report = self_check(corpus, emit=args.emit)
        print("self-check OK: corpus valid, embedded answers score perfectly, gate refuses degraded variants")
    else:
        if not args.responses:
            parser.error("--responses is required unless --self-check is used")
        responses, notes = load_responses(args.responses, corpus)
        for note in notes:
            print(f"note: {note}", file=sys.stderr)
        report = evaluate(corpus, responses)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)
    return 0 if report["false_merge_authorizations"] == 0 and report["label_accuracy"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
