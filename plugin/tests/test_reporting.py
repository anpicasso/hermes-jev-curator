from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from plugin.candidates import deterministic_relation
from plugin.models import CandidatePair, MergePlan, RelationJudgment, Settings, SkillArtifact
from plugin.reporting import (
    CallRecord,
    RunReport,
    ScanError,
    render_json,
    render_markdown,
    run_paths,
    safe_name,
    to_dict,
    write_report,
)


def artifact(name: str, *, text: str = "", digest: str = "0" * 16, **extra) -> SkillArtifact:
    return SkillArtifact(name=name, path=Path("/skills") / name, description="", text=text, digest=digest, **extra)


def pair(a: str, b: str, similarity: float = 0.5, signals: tuple[str, ...] = ()) -> CandidatePair:
    return CandidatePair(a=a, b=b, a_digest="a" * 16, b_digest="b" * 16, similarity=similarity, signals=signals)


def judgment(a: str, b: str, relation: str = "duplicate", confidence: float = 0.9) -> RelationJudgment:
    return RelationJudgment(
        a=a, b=b, a_digest="a" * 16, b_digest="b" * 16, relation=relation, confidence=confidence,
        probabilities={}, coverage=0.8, preservation_a_in_b=0.7, preservation_b_in_a=0.6, conflict=0.1,
        contract_version="skill-relations-v1", raw_model="jev-latest",
    )


def sample_report(**overrides) -> RunReport:
    base: dict[str, Any] = dict(
        mode="observe",
        started_at="2026-09-20T23:00:00-05:00",
        finished_at="2026-09-20T23:00:30-05:00",
        inventory=(artifact("zeta", digest="z" * 16),
                   artifact("alpha", digest="a" * 16, protected_reasons=("pinned",))),
        candidates=(pair("zeta", "alpha", 0.5, ("name-prefix",)), pair("alpha", "beta", 0.8)),
        judgments=(judgment("alpha", "beta"),),
        plans=(MergePlan(
            plan_id="plan-2", canonical="alpha", canonical_digest="a" * 16, absorbed=("beta",),
            absorbed_digests={"beta": "b" * 16}, relation_keys=("alpha::beta",), status="validated",
            blockers=("digest mismatch",),
        ),),
        blockers=("budget exhausted",),
        errors=(ScanError(stage="judge", kind="RuntimeError", message="Jev request failed"),),
        calls=(CallRecord(
            stage="judge", model="jev-latest", attempts=2, http_status=200, latency_ms=1500.5,
            prompt_tokens=100, completion_tokens=40, total_tokens=140, cost_usd=0.0123,
        ),),
        settings=Settings(),
    )
    base.update(overrides)
    return RunReport(**base)


class ReportingTests(unittest.TestCase):
    def test_stable_order_and_determinism(self):
        forward = sample_report()
        backward = sample_report(
            inventory=tuple(reversed(forward.inventory)),
            candidates=tuple(reversed(forward.candidates)),
            errors=tuple(reversed(forward.errors)),
            calls=tuple(reversed(forward.calls)),
        )
        self.assertEqual(render_markdown(forward), render_markdown(backward))
        self.assertEqual(render_json(forward), render_json(backward))
        data = to_dict(forward)
        self.assertEqual([row["name"] for row in data["inventory"]["skills"]], ["alpha", "zeta"])
        self.assertEqual([row["key"] for row in data["candidates"]["pairs"]], ["alpha::beta", "alpha::zeta"])
        self.assertEqual(data["candidates"]["pairs"][0]["baseline"],
                         deterministic_relation(pair("alpha", "beta", 0.8)))
        md = render_markdown(forward)
        self.assertLess(md.index("alpha::beta"), md.index("alpha::zeta"))

    def test_candidate_dedup_by_key(self):
        report = sample_report(candidates=(pair("alpha", "beta", 0.8), pair("beta", "alpha", 0.7)))
        self.assertEqual(to_dict(report)["candidates"]["count"], 1)

    def test_bodies_and_secrets_are_never_rendered(self):
        body_marker = "BODY-MARKER-DO-NOT-RENDER"
        report = sample_report(
            inventory=(artifact("leaky", text=body_marker),),
            blockers=("token=zzz-super-secret",),
            errors=(ScanError(
                stage="judge", kind="RuntimeError",
                message="auth failed: Bearer abcdefgh12345678 sk-live-abcdef123456 password=hunter2",
            ),),
        )
        for rendered in (render_markdown(report), render_json(report)):
            self.assertNotIn(body_marker, rendered)
            for fragment in ("abcdefgh12345678", "sk-live", "hunter2", "zzz-super-secret"):
                self.assertNotIn(fragment, rendered)
            self.assertIn("[redacted]", rendered)
        json.loads(render_json(report))

    def test_write_report_atomically_into_supplied_directory(self):
        report = sample_report()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "nested" / "run-1"
            report_path, json_path = write_report(report, run_dir)
            self.assertEqual((report_path.name, json_path.name), ("report.md", "run.json"))
            self.assertEqual(sorted(entry.name for entry in run_dir.iterdir()), ["report.md", "run.json"])
            first = report_path.read_text(encoding="utf-8")
            write_report(report, run_dir)
            self.assertEqual(report_path.read_text(encoding="utf-8"), first)
            self.assertEqual(sorted(entry.name for entry in run_dir.iterdir()), ["report.md", "run.json"])
            self.assertTrue(json.loads(json_path.read_text(encoding="utf-8"))["read_only"])

    def test_all_host_strings_are_redacted_and_symlinked_directory_is_refused(self):
        report = sample_report(
            inventory=(artifact("alpha", provenance="token=zzz-super-secret"),),
            judgments=(RelationJudgment(
                a="alpha", b="beta", a_digest="a", b_digest="b", relation="unrelated",
                confidence=1.0, probabilities={}, coverage=1.0,
                preservation_a_in_b=0.0, preservation_b_in_a=0.0, conflict=0.0,
                contract_version="skill-relations-v1", raw_model="sk-live-abcdefghijk"),),
        )
        rendered = render_json(report) + render_markdown(report)
        self.assertNotIn("zzz-super-secret", rendered)
        self.assertNotIn("sk-live-abcdefghijk", rendered)
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside"
            outside.mkdir()
            link = Path(tmp) / "run-link"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                write_report(report, link)

    def test_unsafe_filenames_are_rejected(self):
        for bad in ("", "..", "../x", "a/b", ".hidden", "a\\b", "a b"):
            with self.assertRaises(ValueError):
                safe_name(bad)
        self.assertEqual(safe_name("report.md"), "report.md")
        self.assertEqual(run_paths(Path("/tmp/run"))[1].name, "run.json")


    def test_modes_recorded_and_report_is_read_only(self):
        for mode in ("observe", "guard", "apply"):
            data = to_dict(sample_report(mode=mode))
            self.assertEqual(data["mode"], mode)
            self.assertTrue(data["read_only"])
            self.assertIn(f"# Jev curator run — {mode}", render_markdown(sample_report(mode=mode)))
        self.assertEqual(to_dict(sample_report(mode="nonsense"))["mode"], "observe")

    def test_latency_token_and_cost_inputs_aggregate(self):
        report = sample_report(calls=(
            CallRecord(stage="judge", model="jev-latest", attempts=1, http_status=200, latency_ms=100.0,
                       prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.001),
            CallRecord(stage="verify", model="jev-latest", attempts=3, http_status=529, latency_ms=250.25,
                       prompt_tokens=20, completion_tokens=10, total_tokens=30, cost_usd=0.002),
        ))
        summary = to_dict(report)["summary"]
        self.assertEqual(summary["calls"], 2)
        self.assertAlmostEqual(summary["latency_ms"], 350.25)
        self.assertEqual((summary["prompt_tokens"], summary["completion_tokens"], summary["total_tokens"]),
                         (30, 15, 45))
        self.assertAlmostEqual(summary["cost_usd"], 0.003)

    def test_call_record_from_response_and_finite_json(self):
        class Response:
            model = "jev-latest"
            attempts = 2
            http_status = 200
            usage = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16, "cost": 0.005}

        record = CallRecord.from_response(Response(), stage="judge", latency_ms=99.5)
        self.assertEqual((record.prompt_tokens, record.completion_tokens, record.total_tokens), (12, 4, 16))
        self.assertEqual((record.attempts, record.http_status), (2, 200))
        self.assertAlmostEqual(record.cost_usd, 0.005)
        poisoned = CallRecord(stage="judge", latency_ms=float("inf"), cost_usd=float("nan"), attempts=-3)
        self.assertEqual((poisoned.latency_ms, poisoned.cost_usd, poisoned.attempts), (0.0, 0.0, 0))
        json.loads(render_json(sample_report(calls=(poisoned,))))

    def test_empty_report_and_plan_details(self):
        empty = RunReport()
        md = render_markdown(empty)
        self.assertIn("## Summary", md)
        self.assertNotIn("## Candidates", md)
        self.assertNotIn("## Blockers", md)
        data = json.loads(render_json(empty))
        self.assertEqual(data["summary"]["inventory"], 0)
        self.assertEqual(data["summary"]["calls"], 0)
        self.assertIsNone(data["settings"])

        full = to_dict(sample_report())
        self.assertEqual(full["plans"]["applicable"], 0)
        self.assertEqual(full["plans"]["items"][0]["relation_keys"], ["alpha::beta"])
        self.assertEqual(full["blockers"], ["budget exhausted"])
        rendered = render_markdown(sample_report())
        self.assertIn("digest mismatch", rendered)
        self.assertIn("budget exhausted", rendered)


if __name__ == "__main__":
    unittest.main()
