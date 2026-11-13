# How we used CuratorKIT in the data pipeline (SFT v2)

Written 2026-11-10, numbers refreshed 2026-11-12 after the rebuild described under "Rebuild history". Everything below was run and checked unless it is under
"Not used / not verified". Numbers come from `data/sft_v2/curatorkit/manifest.json` and `data/sft_v2/STATS.md`.

## What CuratorKIT is doing here

CuratorKIT is the curation and audit layer between data generation and training:

```
generators (ours)                         CuratorKIT 0.1.24                       trainer-ready files
augment_train.py  ----\
gen_schema_first.py ---+--> curatorkit_input.jsonl --> SchemaGate -> ExactDedup -> MinHashDedup -> SecretsGate --> survivors
convert_paraloq.py ---/                                                                          |
                                                                                                 v
                                                  manifest.json, rejected.jsonl, lexsi_provenance.json,   train_v2.jsonl
                                                  dataset_card.md, checksums.txt                          sft_messages_train.jsonl
```

It does not generate our data. Our generators do, because the gold must be exact by construction (AGENTS.md) and CuratorKIT's
generation tasks (`qa`, `preference`, `grpo`, `multiturn`, `evol`, `cot`, `adversarial_*`) are not schema-conditioned extraction.

## Environment

- `curatorkit==0.1.24`, Python 3.12.3. Install: `pip install curatorkit detect-secrets`
  (SecretsGate raises `ImportError: detect-secrets is not installed. Install with: pip install 'curatorkit[hygiene]'` without it).
- No LLM, no API key, no GPU. `token_stats` in the manifest is all zeros.
- Wall clock for about 3000 rows: 184 s to 192 s. SecretsGate is the slow stage (about 10 to 30 rows per second).

## Input format

One JSONL row per training example, written by `data/build_sft_v2.py`:

| CuratorKIT field | What we put in it |
|---|---|
| `id` | our row id (unique across sources; the script asserts this) |
| `instruction` | the frozen system prompt |
| `input` | the frozen user message: `<JSON_SCHEMA>...</JSON_SCHEMA>` + `<DOCUMENT>...</DOCUMENT>` |
| `output` | gold JSON, `ensure_ascii=False, sort_keys=True` (same as the SFT notebook) |
| `task_type` | `instruction_following` |
| `metadata` | `{"domain": ..., "source": ...}` |

## Configuration used

```python
from curatorkit import Curator, CuratorConfig
cfg = CuratorConfig(
    dataset="data/sft_v2/curatorkit_input.jsonl",
    dedup="minhash", minhash_threshold=0.85,
    clean=False,
    schema_gate=True, min_tokens=10, max_tokens=3500,
    secrets_gate=True,
    export_formats=["alpaca"],
    output_dir="data/sft_v2/curatorkit", name="sft_v2",
)
res = Curator(cfg).run()
kept_ids = [s.id for s in res.passed]      # survivors keep their id; we use it to fetch our own record
```

Pipeline config hash (identical in both builds, the configuration did not change): `85e8cd842d63fd90`.

## What each stage did (this run)

| Stage | In | Out | Note |
|---|---|---|---|
| JSONLReader | 2994 | 2994 | pool = 1517 augmented + 1232 schema-first + 220 paraloq + 25 unaugmented originals |
| SchemaGate | 2994 | 2994 | 0 rejected (non-empty fields, token bounds) |
| ExactDeduplicator | 2994 | 2994 | 0 removed |
| MinHashDeduplicator (0.85) | 2994 | 2854 | **140 near-duplicates removed** |
| SecretsGate | 2854 | 2832 | **22 rejected**: 17 hex high-entropy, 4 base64 high-entropy, 1 both |
| AlpacaExporter | 2832 | 2832 | `sft_alpaca.jsonl` (not used for training, see below) |

Our own checks run after CuratorKIT (not CuratorKIT stages), in this order: a **split guard** that drops any row whose schema or field-name set also appears
in `val`, `test` or the FieldBench files (0 dropped in the current build; see "Rebuild history"), `jsonschema` re-validation of every survivor (0 dropped),
and a length cap of 2300 estimated tokens (72 dropped). Final dataset: **2760 rows**: augmented 1411, schema-first 1176, paraloq 151, originals 22.

## Decisions that differ from the defaults, and why

1. **`clean=False`.** The text cleaner normalises whitespace and unicode in the text fields. Gold exactness depends on the
   document text, so we did not let it rewrite documents. We did not measure what it would have changed; this was a precaution.
2. **We do not train from CuratorKIT's `messages` export.** In a 180-row test, the exported row we inspected had only `user` and
   `assistant` turns, with the system prompt text at the start of the user turn (we passed it as `instruction`). Our prompt format is frozen (separate system role),
   so `sft_messages_train.jsonl` is written by our script from the surviving ids, in the frozen format.
3. **Order of the pool matters for MinHash.** The first row of a near-duplicate pair survives. We pooled paraloq, then
   schema-first, then augmented, then originals. Originals that have an augmented copy are excluded from the pool on purpose,
   because the two are near-duplicates by construction and we want the augmented one.
4. **SecretsGate is on, but it produces false positives on synthetic data.** The 22 rejections are codes, UUIDs and ids in
   synthetic documents (for example `paraloq_MedicalLaboratoryResults_16`), not real secrets. 0.8% of rows; we left the
   thresholds at their defaults (`secrets_hex_limit=3.0`, `secrets_base64_limit=4.5`). The options exist in `CuratorConfig`
   if someone wants to loosen them.
5. **PII pseudonymisation is off.** All data is synthetic, and replacing values would break exact gold.
6. **Which fields MinHash compares is not something we checked.** Documents from the same template with different values did
   not collapse into each other (only 140 of 2994 removed in the current build). We did not measure removals per source in the rebuild;
   in the first build the augmented rows lost the most (112 including secrets rejections), as expected for variants of each other's templates.

## Audit trail (commit these, they are small)

`data/sft_v2/curatorkit/`: `manifest.json` (stage counts, rejection breakdown, versions), `rejected.jsonl` (22 rows with
`rejecting_step` and `rejection_reason`), `lexsi_provenance.json` (`lexsi.provenance/1`), `dataset_card.md`, `README.md`,
`checksums.txt` (SHA-256 of `rejected.jsonl`, `sft_alpaca.jsonl`, `lexsi_provenance.json` and `manifest.json`), and `sft_alpaca.jsonl`.
`sft_alpaca.jsonl` is large (9 MB); the others are under 200 KB. `lexsi_provenance.json` records `generator_model: null` and `judge_model: null`, which is correct:
no LLM was involved in generation or filtering.

## Rebuild history (why the numbers changed, and why the pipeline itself did not)

- **First build, 2026-11-10:** from `aya-json-sft-dataset` at commit `0004dbe1` (train 1554). 3006 pooled, 2841 after CuratorKIT, 2769 final.
- **What happened:** Sujal pushed a new revision the same day (`e26e691b`, 14:29 UTC, train 1542). He moved rows that had been in the older train into the new
  `val` and `test`. Our first build was therefore contaminated against the current splits (88 of 120 `val` and 84 of 120 `test` rows shared a field-name set
  with our training rows). The verifier caught it: the document ids of an earlier result file were missing from the current dataset.
- **Rebuild, 2026-11-12:** the same four scripts re-run, with the dataset pinned to `e26e691b` (`data/aya-json-sft-dataset/REVISION.txt`, also printed in `STATS.md`).
  CuratorKIT ran again with the same configuration; its manifest, rejection log, provenance and checksums were regenerated (counts in the table above).
  The schema-first (JSONSchemaBench) and paraloq inputs did not change, and neither did their share of the secrets rejections (22 both times).
- **Split guard added** (our code, after CuratorKIT): the build now drops any training row whose canonical schema or field-name set also appears in `val`, `test`,
  `test_b` or `test_c`, and reports the number in `STATS.md`. In the rebuild it dropped 0 because the base is now the current train.
- **What this means for the submission:** CuratorKIT is still the curation and audit layer of the pipeline; Sujal's dataset is one input to it (the augmented rows are
  derived from his train), not a replacement for it. Pin the dataset revision in the write-up, because his repo can change again.

## Not used / not verified (do not claim these in the submission)

- **LLM-judge gates** (`HallucinationGate`, `RewardGate`) and adaptive recovery: need an LLM endpoint; not configured.
  Grounding is checked by our own deterministic code in the generators instead (a value must appear in the document or in the
  rendering the script produced).
- **DiversityGate / embedding dedup:** needs an embedding model download; not run.
- **Custom gates and `preprocessing_fn`:** `preprocessing_fn` exists in the `CuratorConfig` signature, but we did not use or
  test it. The README says custom gates are plugin points and points to `CONTRIBUTING.md` and `docs/reference/architecture.md`,
  which we did not read.
- **Generation tasks:** not applicable (see top).
- **Sujal's earlier notebook** ran `Curator(...)` (MinHash, clean, alpaca/sharegpt export) and then did its own stage-by-stage
  filtering in plain Python (length gate, shingle dedup, `jsonschema`, grounding), as far as its code shows. This document
  describes the SFT v2 build, not that run.

## Reproduce

```bash
pip install curatorkit detect-secrets faker jsonschema pandas pyarrow huggingface_hub
python data/augment_train.py        # needs data/aya-json-sft-dataset/train.jsonl (snapshot_download of the HF dataset)
python data/gen_schema_first.py     # downloads epfl-dlab/JSONSchemaBench train splits
python data/convert_paraloq.py      # downloads paraloq/json_data_extraction
python data/build_sft_v2.py --eval-extra <fieldbench test_b.jsonl> <fieldbench test_c.jsonl>
```
All steps are seeded (`--seed 42`). Outputs: `data/sft_v2/train_v2.jsonl`, `sft_messages_train.jsonl`, `STATS.md`, `curatorkit/`.

## Paragraph for the write-up (edit freely)

> We used CuratorKIT 0.1.24 as the curation and audit layer of our data pipeline. Training examples are generated
> schema-first by our own code, so the gold is exact by construction. We ingest the pooled JSONL into CuratorKIT, which applies its
> schema/length gate, exact and MinHash (0.85) near-duplicate removal and secrets detection, and writes the manifest, rejection log and
> provenance record. Of 2994 pooled examples, 140 were removed as near-duplicates and 22 by the secrets gate (false positives on
> synthetic identifiers). We kept CuratorKIT's text cleaning off, to leave documents byte-exact under their gold values, and wrote the
> final chat-format file ourselves to keep our frozen prompt format. Judge-based gates and embedding diversity filtering were not used.
