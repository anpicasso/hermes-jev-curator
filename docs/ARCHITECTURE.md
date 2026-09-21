# Architecture — hermes-jev-curator

Visual companion: [interactive architecture diagram](architecture.html).

Design and implementation notes for the 0.1.0 experimental release. Status is marked per
component: **implemented** (in `plugin/`, unit-tested), **verified** (exercised through an
installed snapshot), or **not yet done** (no production-library mutation or model-quality
benchmark).

Verified against Hermes Agent v0.21.3 using source checkout `522e121e`: `hermes plugins doctor` and
`hermes plugins validate` pass, the plugin loads through the real discovery path, and its
registered surfaces were probed under a scratch `HERMES_HOME`. The installed default-profile
snapshot was also exercised against TypeSafe with whole, chunked, and locally unavailable pairs.

## Pipeline

    inventory ─► deterministic candidates ─► Jev typed judgments ─► direct-edge graph ─► plan ─► apply (ledgered skill_manage)

### 1. Inventory and candidate generation — implemented

`plugin/inventory.py` walks `$HERMES_HOME/skills` through core's own reporting
(`tools.skill_usage.curated_report()` for curator-managed skills, `usage_report()` when
unmanaged skills are requested) and builds one `SkillArtifact` per skill package: name,
description, bounded package text, a SHA-256 package digest, provenance, usage counters, and
`protected_reasons`. Protection is inherited from core, never re-listed: `pinned`, `bundled`,
`hub`, `external`, `disabled`, `essential`, `org-mirror`, `cron-referenced`, `core-ineligible`
(via `skill_usage.is_curation_eligible`), `eligibility-unknown`, plus package-level flags
(`contains-symlink`, `contains-binary`, `unreadable-package`, `package-too-large`,
`text-too-large`, `contains-non-utf8`, `unreadable-entry`). Symlinks are
never followed; text is capped per file and per package; hashing has its own byte budget.

`plugin/candidates.py` then produces `CandidatePair`s with no model call:

- tokenization over name, description, and body; weighted cosine plus Jaccard terms, with a
  same-prefix bonus;
- pairs kept above a minimum similarity or on the `name-prefix` signal; top-k neighbors per
  skill; dedup by sorted name pair; stable final order; capped by `max_pairs`.

Determinism is the point: identical inputs must yield identical pairs, ids, and order, or every
stored plan and cache entry goes stale on the next run. `deterministic_relation()` is an ablation
baseline only — it proposes, it never authorizes a mutation, and it is carried through reports
next to every Jev judgment so the two can be compared.

### 2. Jev typed judgments — implemented and live-exercised

`plugin/questions.py` freezes contract `skill-relations-v3`: one `relation` choice with eight
criteria (`duplicate`, `a_subset_of_b`, `b_subset_of_a`, `same_class`, `complementary`,
`conflict`, `unrelated`, `insufficient_evidence`), plus typed coverage, containment, conflict,
and same-class nouls. Pair state is never truncated.

If both redacted packages fit the 160,000-byte serialized-state ceiling and the conservative
24,000-token estimate, one whole-pair request asks the unchanged six-question contract. Serialized
size includes JSON escaping. The stdlib-only token estimate intentionally over-counts code,
punctuation, newlines, and high-entropy runs; it exists because a live 158k Markdown/code state was
byte-safe but exceeded the model context.

Otherwise the planner evaluates each plannable containment direction separately: the candidate
being absorbed is split at package-file markers, then Markdown headings, then fixed-overlap hard
boundaries, while the proposed containing side travels whole in every request. Every assembled
state is rechecked against both ceilings. A chunk request asks only relation, coverage, conflict,
and the matching containment noul. If neither direction can keep its containing side whole, the
pair gets local `insufficient_evidence` with `evidence="unavailable"` and no network call.

Aggregation is deterministic and fail-closed: every planned chunk must answer; preservation and
coverage use `min`, conflict uses `max`, and any conflict label wins. Certified directions produce
duplicate or directional containment labels; an unmeasured direction is always `0.0`, so the
graph's existing preservation gate refuses it. No model output or summary crosses calls.

`plugin/transport.py` sends `{state, model, questions}` and validates everything that comes
back:

- every asked question must be answered — a missing or malformed answer is an error, never a
  default (no `… or 0.0` on a renamed key, no benign branch for an absent choice);
- nouls and confidences: finite, `[0, 1]`, bools rejected;
- choice: must be one of the requested criteria and must be the argmax of `probabilities`
  (which must match the option set exactly and sum to 1 within tolerance);
- `score` (rubric questions): numeric, finite, inside the declared range;
- unknown question types are rejected; unknown extra answer keys cannot introduce decision keys.

Response handling: request/response byte caps, one overall deadline for up to three attempts,
retry on 429/529/5xx, same-origin-only redirect policy. Credentials resolve through
`resolve_runtime_provider` (provider pool) then `get_env_value_prefer_dotenv` (key_env); state
passes through core's redactor before egress; a missing/failed redactor refuses the request.

`plugin/engine.py` walks candidate pairs in rank order and charges `max_requests` by planned Jev
requests, not pairs. A pair that cannot fit the remaining request budget is reported in
`scan["skipped"]` and is never partially judged; later cache hits and cheaper pairs may still be
used. Requests within one pair are sequential, while different pairs keep the four-worker pool.
Judgments are cached in `relations.json` under both content digests, contract version, and judge
model; the v3 contract therefore cannot replay truncation-era v1/v2 rows. Any chunk failure is one
pair error with no partial judgment, and the scan reports `ok: false`.

Live integration smoke test (not a quality benchmark): the installed guard-mode profile inventoried
27 managed skills and produced 100 candidates. With `max_requests: 50`, it returned 9 judgments
(8 chunked, 1 locally unavailable), persisted 91 request-budget skips, reported zero errors, and
authorized no plans. A representative long Markdown/code pair that previously hit the endpoint's
token limit completed after the v3 byte-plus-token planner shipped.

### 3. Direct-edge graph — implemented

`plugin/graph.py` evaluates each judgment exactly once into an `Edge` bound to both content
digests, then builds star-shaped plans:

- **Direct, never transitive.** Every absorbed member carries its own high-confidence
  containment or duplicate judgment against the plan's canonical; A~B plus B~C never merges C
  into A. This is tested.
- **Versioned evidence policy in code** (`skill-graph-v1`): minimum confidence in the absorbing
  relation, minimum coverage, maximum tolerated conflict, minimum preservation of the absorbed
  content in the canonical. Thresholds are not operator-configurable — tune them with
  measurements, not settings.
- **Refusals are explicit and counted**: self-pairs, unknown artifacts, stale hashes, malformed
  evidence, `insufficient_evidence`, `conflict`, low confidence, low coverage, conflict score,
  low preservation. The graph reports refusal counts per reason; chunked evidence can authorize
  only containment measured from every planned chunk while the containing side was whole.
- **Blockers stop a plan**: protected members (with the core-derived reason list), any conflict
  inside the group, or a canonical that is itself absorbed elsewhere. A plan is `validated`
  only with zero blockers; `MergePlan.applicable` requires exactly that.
- **Content-addressed ids**: `merge-<sha256[12]>` over graph version, canonical, digests, and
  relation keys; when nothing authorizes, an explicit `noop-…` plan records the graph state
  instead of an empty list.
- **Deterministic canonical choice**: most-used skill first, then name; duplicate members are
  assigned to exactly one claimant so no skill is archived by two plans.

### 4. Apply — implemented, gated, never run against a live library

`CuratorEngine.apply()` only archives sources that a validated plan says the canonical already
preserves. It refuses unless every gate passes:

1. `mode: apply` **and** an explicit `--apply` on the terminal command (chat can never apply);
2. every selected plan carries its canonical digest, every absorbed digest, and a direct
   canonical-member relation edge;
3. the plan set is free of overlaps and consolidation chains;
4. `skills.write_approval` is disabled, because staged replay does not preserve the
   background-review provenance that selects recoverable archive behavior;
5. every affected package digest still matches the judgment (`digests_match` re-collected at
   apply time — a changed skill means "rescan required");
6. a pre-apply skills snapshot (`agent.curator_backup.snapshot_skills`) succeeds — a missing
   snapshot refuses the destructive path;
7. every target is unprotected and its frontmatter name exactly matches its package directory;
   mismatches cannot enter a plan because core's archive surface cannot address them safely;
8. mutations go through `ctx.dispatch_tool("skill_manage", …)` with the write origin set to
   `background_review`, i.e. core's own guards and the skill ledger stay in charge, and success
   requires core's explicit `_archived: true` result;
9. after each archive, targeted inventory verifies the source is gone and the canonical digest
   is unchanged; a failure stops immediately and returns the snapshot plus restore commands for
   sources already archived.

Core's curator still writes the umbrella prose. The plugin never writes skill text; it decides
what is safe to absorb and executes only the archival half, through the ledgered surface.

## Seams into stock Hermes

All seams are stock plugin APIs — no core edits, no monkeypatching.

| seam | how it is used | verification |
|---|---|---|
| manifest | `kind: standalone`, `manifest_version: 2`, list-form `provides_tools` / `provides_hooks`; `validate` diffs the declaration against a recording run of `register(ctx)` | `hermes plugins validate plugin` passes, including the capability probe and security scan |
| tool | `jev_skill_relations` registered with `toolset="skills"` so it merges into the built-in skills toolset the curator fork sees | `get_toolset("skills")` includes it; a no-network call returned a bounded JSON payload |
| system prompt | `register_system_prompt_section` with a callable gated on `session_info["platform"] == "curator"` — the curator contract reaches the fork and no other surface | rendered for `platform="curator"`, absent for `platform="cli"` |
| hooks | `on_skill_lifecycle` appends facts; `pre_tool_call` gates only background-review `skill_manage` in `guard`/`apply`, using local hash-bound plans and no network | both callbacks present after real discovery; foreground and observe mode stay inert in tests |
| commands | `register_cli_command` (`hermes jev-curator …`) and `register_command` (`/jev-curator`), both parsed by `plugin/commands.py` | `commands_registered: ['jev-curator']` on real load |
| config | `ctx.get_config(...)` → `plugins.entries.jev-curator.settings.*`, re-read per call. The model setting is named `jev_model` because `model` is a host-reserved root | probed: `mode`/`provider`/`base_url`/`key_env`/`top_k`/`jev_model` survive |
| state | `$HERMES_HOME/jev-curator/`: `audit.jsonl` (0600, one rotation, redacted, byte-bounded), `state.json`, `relations.json` cache, `guard_plans.json`, `claim.lock` (O_EXCL, dead-owner recovery), `reports/*.json`. It always follows the active profile and never lives inside `skills/` | path resolution and core-file refusal are unit-tested |
| execution | `ctx.dispatch_tool("skill_manage", …)` for the archival mutation, write origin `background_review` | registry dispatch does **not** run `pre_tool_call` hooks, so apply repeats the same digest/edge/protection gates before snapshot and dispatch |
| unload | registration handles; no `atexit` | unloading the plugin removed its tool, hook, section, and command |

Notes that matter:

- Hooks and tool handlers must accept `**kwargs`; Hermes adds payload fields over time and
  inspects signatures. Handlers here return JSON/text and never raise.
- Package imports stay inside handlers, so `register(ctx)` runs in the bare validation probe
  (no host state, no network at import time).
- `hermes plugins doctor` proves `register(ctx)` returned, not that a hook fires; verify through
  the real discovery path (`discover_plugins(force=True)` + `get_plugin_manager()`).

## Safety invariants

1. **Read-only by default.** `observe` is the default mode; the tool surface (`jev_skill_relations`)
   never mutates, and nothing in `plugin/` writes a skill file.
2. **Curated population only.** Inventory is filtered through core's own predicates; bundled,
   hub, external, pinned, essential, cron-referenced, and frontmatter-name/package-directory
   mismatched skills carry explicit protection reasons, and a protected member blocks its whole
   plan. A copied allowlist is forbidden — it would drift from core and eventually merge a
   protected skill away.
3. **No deletion.** The apply path archives sources (`skill_manage delete` with
   `absorbed_into`), which core implements as a recoverable move to `skills/.archive/`; there is
   no purge path.
4. **Typed, complete evidence.** A missing or malformed whole-pair answer or chunk fails that
   pair; no partial answer set is aggregated. Chunked evidence can authorize only a containment
   direction measured across every planned chunk while the containing side stayed whole. There
   is no truncation path.
5. **Hash binding.** Pairs, cache entries, and plan ids pin content digests; apply re-checks
   every affected package and refuses on any change. Mtime-only changes do not matter
   (content-addressed, not timestamp-addressed).
6. **Direct edges only.** No transitive closure, no inferred relations, no similarity score may
   authorize absorption — only a judged, direct, high-confidence containment or duplicate.
7. **Explicit refusals.** Every refusal reason is recorded and counted (graph `refusals`, scan
   `errors`, request-budget `skipped`), so a blocked or deferred pair is diagnosable instead of
   silent.
8. **Apply gates.** `mode: apply` plus a terminal `--apply` plus validated plans plus a
   snapshot plus digest/name/protection checks plus post-archive verification; chat can never
   apply.
9. **Guard scope.** Relation evidence may authorize only an exact hash-bound absorption.
   Patches, edits, removals, and overwrites are blocked because a relation does not prove that
   proposed new bytes preserve every rule. Foreground user calls and harmless creates are inert.
10. **Egress boundary.** HTTPS, default port, no URL credentials/query/fragment, same-origin
   redirects only, byte caps, bounded retries under one deadline, fail-closed redaction; no
   request at all unless `mode` is not `off` and `allow_content_egress: true`. The endpoint is always the configured one — no silent
   fallback to a default host.
11. **State hygiene.** 0600 files, redacted and bounded audit rows, atomic writes, refusal to
    write core-owned files (`skills/.usage.json`, `skills/.curator_state`).
12. **Mode safety.** Unknown or malformed `mode` falls back to `observe`, never to `apply`.
13. **Core untouched.** Everything rides stock seams; no core file is modified.

## Failure semantics

- **Transport:** bounded attempts inside one `timeout_seconds` deadline; retry only 429/529/5xx
  and connection-level errors; other 4xx are terminal and surfaced with the offending field.
- **Contract:** any asked question unanswered or malformed, or any chunk request failing → one
  pair error naming the question/request; no partial judgment is emitted, the rest of the scan
  continues, and the scan reports `ok: false`. A pair over the remaining request budget is
  reported in `scan["skipped"]` and receives no partial request set.
- **Apply:** stops on the first failed mutation and reports the snapshot, what was applied, and
  exact `hermes curator restore <name>` commands for already-archived sources.
- **Audit/state writes** never change a verdict (best-effort, DEBUG-logged failures).

## Roadmap and kill criteria

### Delivered — wire-up and bounded long-pair evidence

Manifest + registration, `jev_skill_relations` tool in the skills toolset, curator-gated prompt
section, lifecycle observer, offline `pre_tool_call` mutation guard, `hermes jev-curator` /
`/jev-curator` commands, graph + plans, state/cache/audit/reports, gated `run --apply`, and the v3
whole/chunked/unavailable evidence planner. Unit suite in `plugin/tests/`; `doctor` + `validate`
pass; the live TypeSafe route and an installed real-library scan have been exercised.

Remaining before treating it as production-proven:

- measure false blocks and useful catches from the offline `pre_tool_call` guard over repeated
  real-library runs;
- benchmark relation quality against a reviewed corpus; the live scan proves integration, not
  semantic accuracy;
- exercise `run --apply` only in a disposable profile before any production-library mutation.

Kill criteria:

- the typed judgments do not beat or meaningfully differ from the deterministic baseline on real
  candidate pairs → the Jev layer adds cost without signal;
- observe reports contain no pair the core curator's own pass would not have considered → the
  added layer has no demonstrated value;
- a single false-positive absorption on a realistic corpus → `apply` goes back behind an
  explicit opt-in and the graph policy is re-derived from measurements.

### Next — generated-umbrella preservation verification

Extend the current relation guard with preservation re-verification against *generated* umbrella
bytes before sources are archived. The current relation plan deliberately blocks those content
writes because pairwise evidence cannot certify unseen replacement text.

Kill criteria: if the hook's local decision needs a network call, kill it — `pre_tool_call` is
the only fail-closed hook and a Jev outage must not block every curator write. If guard and
apply verdicts ever disagree on the same corpus (two implementations), kill the guard half.

### Later — beyond pairs

Component-level plans from the direct-edge graph, stale/unused signals from usage telemetry, and
possibly utility evidence.

Kill criteria: if per-skill causal measurement (ASSAY-style) is unaffordable on the operator's
traffic, or adds no signal over existing usage telemetry, do not build it.

## Prior art

- **MemRefine** — [arXiv:2606.13177](https://arxiv.org/abs/2606.13177). Similarity proposes
  candidate pairs; an LLM judge decides delete/merge/preserve on factual content. The closest
  published description of this plugin's pair pipeline; the differences here are a fixed typed
  contract instead of free-form judge output, and no storage-budget loop.
- **Mem0** — [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0). Production memory layer
  with a compression engine and LoCoMo/LongMemEval-style evaluation. The ecosystem baseline for
  "memory curation as infrastructure"; memory-store oriented, not skill-library oriented.
- **ASSAY** — [arXiv:2606.15390](https://arxiv.org/abs/2606.15390). Measures per-skill causal
  contribution by randomized masking, and shows LLM-judgment-only curation conflates generation
  with curation. The strongest argument for a v3 evidence stage — and against trusting typed
  judgments alone.
- **SkillBrew** — [arXiv:2605.29440](https://arxiv.org/abs/2605.29440),
  [github.com/Applied-Machine-Learning-Lab/EMNLP26_SkillBrew](https://github.com/Applied-Machine-Learning-Lab/EMNLP26_SkillBrew).
  Bank-level, multi-objective (utility/diversity/coverage) curation with a propose-then-verify
  loop. The template for v3's component-level plans; skipped in v1 because it needs rollout
  evidence this project does not have yet.
- **Letta / LangMem** — [github.com/letta-ai/letta](https://github.com/letta-ai/letta),
  [github.com/langchain-ai/langmem](https://github.com/langchain-ai/langmem). Background
  consolidation precedents ("sleep-time" agents, background memory managers) with the same shape
  as Hermes' curator pass: a background fork rewrites stored knowledge. Neither adds a typed
  per-pair decision layer.
- **caura-memclaw / Caura** — [github.com/caura-ai/caura-memclaw](https://github.com/caura-ai/caura-memclaw).
  Governed multi-agent shared memory with trust tiers and audit trails, MCP-native. Ecosystem
  precedent for governance-first curation (who may write, what is audited) rather than a
  technique this project borrows.
