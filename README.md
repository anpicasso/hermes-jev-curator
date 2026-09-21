# hermes-jev-curator

**Typed, per-pair skill-relationship judgments for [Hermes Agent](https://github.com/NousResearch/hermes-agent)'s
background skill curator, served by [TypeSafe's](https://typesafe.ai) Jev decision model.**

> **Proof of concept (0.1.0).** The plugin loads, registers, and runs against stock Hermes
> (`hermes plugins doctor` / `validate` pass); its behavior is covered by the unit suite in
> `plugin/tests/`. It has **not** been exercised against a live Jev endpoint; the bundled
> synthetic corpus is a contract self-check, not a model-quality benchmark, and no live metric
> is claimed. Treat it as experimental: default mode is read-only
> `observe`, and the mutation path (`run --apply`) refuses unless `mode: apply` is set
> explicitly.

## What it does

Hermes' background curator reviews agent-created skills on an idle timer: it marks stale skills,
archives idle ones, and — with the LLM pass enabled — rewrites overlapping skills into umbrellas
through the ledgered `skill_manage` tool. That pass decides in prose.

This plugin adds a typed layer in front of it:

1. **Deterministic candidates.** A read-only inventory of curator-managed skills plus lexical
   top-k neighbor pairs. No model call; unchanged inputs produce identical pairs, ids, and order.
2. **Typed judgments.** Each candidate pair uses one bounded request when both packages fit, or
   deterministic per-direction chunk requests otherwise. Every request answers a versioned
   question contract: an 8-way `relation` choice (`duplicate`, `a_subset_of_b`, `b_subset_of_a`,
   `same_class`, `complementary`, `conflict`, `unrelated`, `insufficient_evidence`) plus the
   applicable `coverage`, preservation, and `conflict` probability questions.
3. **Direct-edge graph.** Judgments become typed edges between skill names, bound to both
   content digests. Merge plans are star-shaped — every absorbed member needs its own direct
   containment/duplicate judgment against the canonical; A~B plus B~C never merges C into A.
4. **Core keeps the pen.** The plugin writes no skill text. Its `apply` path only archives
   sources that a validated plan says the canonical already preserves, through core's ledgered
   `skill_manage`, after a pre-apply snapshot. Umbrella prose stays core's job.

## Status (0.1.0 PoC)

| component | state |
|---|---|
| inventory, candidate generation, question contract, transport | implemented, unit-tested |
| plugin manifest + `register(ctx)` (tool / prompt / lifecycle + guard hooks / commands) | implemented; `doctor` + `validate` pass; real-load verified |
| graph, plans, state (audit / cache / lock / reports), `run --apply` | implemented, unit-tested; never run against a live library |
| live Jev endpoint | not yet exercised end-to-end |
| frozen offline benchmark (`benchmarks/`) | 21 synthetic relation/adversarial cases; self-check only, not a live-model quality claim |

## Stock-core integration

The plugin imports stock Hermes modules read-only. Core is not modified, patched, or
monkeypatched; nothing in this tree requires a core change.

| import | used for |
|---|---|
| `tools.skill_usage` | `curated_report()` / `usage_report()` rows, `provenance()`, `is_curation_eligible()` |
| `agent.skill_utils` | `iter_skill_index_files()`, `parse_frontmatter()` |
| `hermes_constants` | `get_hermes_home()` |
| `cron.jobs` | `referenced_skill_names()` — cron-referenced skills are flagged protected |
| `hermes_cli.runtime_provider` | credential resolution through Hermes' provider pool |
| `hermes_cli.config` | `get_env_value_prefer_dotenv()` credential fallback |
| `agent.redact` | mandatory redaction before egress; best-effort redaction in audit/report text |
| `agent.curator_backup` | pre-apply skills snapshot (`snapshot_skills`) |
| `tools.skill_provenance` | marks apply-mode writes as background-review origin |
| `tools.write_approval` | refuses apply when staged replay would lose background-review provenance |

Verified against Hermes Agent v0.21.3 using source checkout `522e121e`.

## Modes

Set under `plugins.entries.jev-curator.settings.mode`; read on every call, so an edit applies
without a restart. Any value that is not one of the four below falls back to `observe` — never
to a write mode.

| mode | behavior today |
|---|---|
| `off` | no Jev requests (inventory, candidates, and the deterministic baseline only) |
| `observe` (default) | judgments + plans + reports; no skill mutation of any kind |
| `guard` | installs current hash-bound relation plans and blocks background destructive `skill_manage` calls that are not an exact authorized absorption; foreground calls remain untouched |
| `apply` | unlocks `hermes jev-curator run --apply` (terminal only): archives sources of complete direct-edge, hash-bound plans via ledgered `skill_manage`, after a snapshot and digest re-check; requires core's `_archived` confirmation and verifies each source disappeared while the canonical stayed unchanged; refuses while `skills.write_approval` is enabled |

## Install

Once the repository is published:

```bash
hermes plugins install anpicasso/hermes-jev-curator/plugin
hermes plugins enable jev-curator
hermes gateway restart
```

For local development, copy `plugin/` to `$HERMES_HOME/plugins/jev-curator/` and add
`jev-curator` to `plugins.enabled`. Plugins are profile-scoped: repeat the install for every
`$HERMES_HOME` that should use it.

## Configuration

Settings live in the host config, never in a file inside the plugin directory (that directory is
a replaceable installed snapshot):

```yaml
plugins:
  enabled: [jev-curator]
  entries:
    jev-curator:
      settings:
        mode: observe          # "off" | observe | guard | apply (quote off for YAML 1.1 parsers)
        provider: typesafe     # typesafe | openrouter | custom
        allow_content_egress: false  # set true only after accepting the disclosure below
        # base_url: https://jev.example/v1/systemone   # required for provider: custom
        # key_env: MY_JEV_KEY    # custom endpoints only
```

| key | default | range | meaning |
|---|---|---|---|
| `mode` | `observe` | off/observe/guard/apply | see [Modes](#modes) |
| `provider` | `typesafe` | typesafe/openrouter/custom | decision-endpoint preset |
| `base_url` | — | https, port 443 | complete Jev decision endpoint; `custom` only |
| `jev_model` | provider default (`jev-latest`; OpenRouter: `~typesafe/jev-latest`) | — | backend model id; renamed because Hermes reserves the plugin setting root `model` |
| `key_env` | — | environment-variable name | custom endpoints only; anonymous when unset |
| `allow_content_egress` | `false` | boolean | explicit consent required before skill text can be sent to any endpoint |
| `timeout_seconds` | `25` | 1–120 | one overall deadline per request, retries included |
| `max_requests` | `50` | 1–500 | per-scan request budget (chunk requests count individually) |
| `max_pairs` | `100` | 1–2000 | per-scan candidate-pair budget |
| `top_k` | `5` | 1–20 | lexical neighbors kept per skill |

### Long pairs

Pairs whose redacted package text fits the measured 160k-byte request-state budget use one
whole-pair request. Larger pairs are split deterministically at package-file markers, then
Markdown headings, then fixed-overlap hard boundaries; one side stays whole while every chunk of
the other is judged. Aggregation is fail-closed (`min` preservation/coverage, `max` conflict): a
missing or malformed chunk fails the pair, and an unmeasured containment direction is `0.0`.
Pairs that cannot keep either containing side whole return `insufficient_evidence` without a
network call. `max_requests` counts actual planned requests; a pair that does not fit the
remaining budget is reported in `scan["skipped"]` and never partially judged.

Endpoint presets:

| provider | endpoint | credential |
|---|---|---|
| `typesafe` (default) | `https://api.typesafe.ai/v1/systemone` | `TYPESAFE_API_KEY` in the active profile's `.env` / environment; a matching installed provider credential is also accepted |
| `openrouter` | `https://openrouter.ai/api/alpha/decisions` | `hermes auth add openrouter`, or `OPENROUTER_API_KEY` |
| `custom` | your `base_url` | only the configured `key_env`; anonymous when unset — it never inherits TypeSafe or OpenRouter credentials |

Credentials resolve through Hermes' own chain (matching-host provider credential → active-profile
`.env` → environment). A credential resolved for another host is never forwarded to the Jev endpoint.
The optional [`jev-approvals`](https://github.com/anpicasso/hermes-jev-approvals) companion
registers the `typesafe-jev` Hermes credential provider; without it, `TYPESAFE_API_KEY` in the
active profile's `.env` or environment is sufficient.

State and audit live under the active profile's `$HERMES_HOME/jev-curator/` (bound the audit log
with `JEV_CURATOR_AUDIT_MAX_BYTES`; `0` disables it). Files are mode `0600`;
audit rows and report text are redacted and bounded; the plugin refuses to write core-owned
files (`skills/.usage.json`, `skills/.curator_state`).

## Commands

```bash
hermes jev-curator status            # mode, inventory size, last run
hermes jev-curator scan              # inventory + candidate pairs (+ Jev judgments)
hermes jev-curator review NAME       # one skill's candidate pairs
hermes jev-curator graph             # edge summary with refusal counts
hermes jev-curator plan              # merge plans and blockers
hermes jev-curator run               # full pipeline, dry; writes a report
hermes jev-curator run --apply       # executes validated plans; needs mode=apply, terminal only
hermes jev-curator doctor            # route / inventory / mutation-default checks
```

Every subcommand takes `--json` for the bounded machine-readable payload. The `/jev-curator`
slash command exposes the same subcommands but always runs dry: `--apply` is refused from chat.

Execution it rides (core's own):

```bash
hermes curator status                 # scheduler state, last run, skill stats
hermes curator run --consolidate      # the LLM umbrella pass this plugin informs
hermes curator ledger                 # per-mutation audit entries
hermes curator rollback [entry-id]    # restore one mutation (or the whole tree)
```

Verify the install:

```bash
hermes plugins doctor plugin --ci     # runtime discovery, manifest parse, import, register
hermes plugins validate plugin        # manifest-vs-registration diff, security scan
python3 -m pytest plugin/tests -q     # unit suite (determinism, graph gates, state, commands,
                                      # transport contract, synthetic merge corpus)
```

## Network egress and data handling

The only network egress is the decision request itself. It is POSTed to the configured endpoint
only when `mode` is not `off` **and** `allow_content_egress: true`. The default is no egress.

- **What leaves the machine after consent:** one bounded state containing redacted skill names and
  either both complete packages or one complete package plus one labeled chunk, plus the question
  contract and configured model id. No package digest is sent; package text is never truncated.
- **Where it goes:** the configured endpoint — `https://api.typesafe.ai/v1/systemone` by
  default, `https://openrouter.ai/api/alpha/decisions` for OpenRouter, or your `custom` URL.
  A third party therefore sees your skill content unless you point `custom` at your own host.
- **Redaction uses Hermes' own redactor** and the request fails closed if that redactor is
  unavailable. Redaction is still a hygiene measure, not a confidentiality control.
- **Transport boundary:** HTTPS on port 443 only; URL credentials, query strings, and fragments
  are rejected; cross-origin redirects are refused; requests and responses are capped at 2 MB;
  retries (429/529/5xx) are bounded and share one overall deadline.

## What it never does

- **The tool never mutates.** `jev_skill_relations` returns judgments only.
- **No direct skill writes.** No code path writes skill files; mutations go through core's
  `skill_manage` so the ledger records them and `hermes curator rollback` can restore them.
- **No deletion.** The apply path archives sources (`delete` with `absorbed_into`), which core
  implements as a recoverable move to `skills/.archive/` (`hermes curator restore <name>`); a
  pre-apply snapshot is taken first, and a missing snapshot refuses the run. A partial failure
  returns the snapshot plus exact `hermes curator restore <name>` recovery commands.
- **No chat-driven mutation.** `/jev-curator run --apply` is refused; apply needs a terminal,
  `mode: apply`, and validated plans whose digests still match.
- **No stale application.** If any affected skill package changed after judgment, apply refuses
  and asks for a rescan. After every archive it re-reads only that source and canonical, requires
  the source to be gone, and requires the canonical digest to remain unchanged.

## Restart

Python plugins load at discovery; there is no hot reload. After installing, enabling, or
changing plugin code, restart the gateway (`hermes gateway restart`; named profiles:
`hermes --profile <name> gateway restart`). Settings are re-read on every call, so config edits
do not need a restart.

## Uninstall

```bash
hermes plugins disable jev-curator    # stop loading it, keep the files
hermes plugins remove jev-curator     # remove the installed snapshot
```

Then delete `$HERMES_HOME/jev-curator/` (audit, cache, state, reports) if you want it gone.
Nothing else to undo: the plugin does not modify skills outside `run --apply`, config, or
credentials, and no core file was changed.

## Detailed documentation

- [Architecture, seams, safety invariants, roadmap, prior art](docs/ARCHITECTURE.md)
- [Interactive architecture diagram](docs/architecture.html)

## Requirements

- Hermes Agent v0.21.2+ (verified against v0.21.3 source checkout `522e121e`)
- Python 3.10+; standard library only — no third-party Python dependencies
- A Jev credential only if you use TypeSafe or OpenRouter

## Licence

MIT.
