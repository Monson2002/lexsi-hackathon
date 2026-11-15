# Teaching Tiny Aya Global to follow any JSON Schema

Write-up for the Lexsi Labs hackathon, Track 1 (fine-tuning and data curation), problem 1 (structured output and extraction).
Every number in this document comes from a result file in this repository (`results/`) or in the Hugging Face repositories listed at the end. Where we did not measure something, this document says so.

## 1. The problem

A user gives the model a **document** and a **JSON Schema**. The model must return one JSON object that:

- is valid JSON and follows the schema (types, enums, required keys, nesting),
- takes every value from the document and never invents a value,
- writes `null` for a value that the document does not contain,
- changes a value (for example a date) only when the schema asks for it.

The goal is the general skill of following arbitrary schemas. A model that learns one fixed form is not useful. So we test on schemas it never saw (`test`), on a domain it never saw (`test_b`), and on real documents (`test_c`).

The prompt format is frozen and identical in training and evaluation: one system message, and one user message with a `<JSON_SCHEMA>` block and a `<DOCUMENT>` block.

## 2. Approach in one picture

```
CuratorKIT:  pool of examples  ->  gates (schema, length, duplicates, secrets)  ->  split guard  ->  2,760 rows
AlignTune:   QLoRA SFT on Tiny Aya Global (loss on the JSON answer only)         ->  SFT v2 adapter
AuditKIT:    base vs adapter on 4 splits, metrics recomputed from raw outputs    ->  verified tables
Extra:       DPO on top of SFT v2 (tested, no gain)    GRPO adapter (earlier, no change)
```

Student model: `CohereLabs/tiny-aya-global` (3.35B parameters). We use all three Lexsi toolkits: CuratorKIT 0.1.24 for data, AlignTune 0.1.33 for training, AuditKIT 1.1.6 for evaluation. Our own code plugs into their flow where a toolkit has no hook (extraction metrics, the split guard, the pair builder).

## 3. Data

### 3.1 Gold is built schema-first
We do not use "an LLM extracts the answer" as gold. We sample a record that satisfies the schema, and then write the document from that record. The gold answer is then exact by construction.

### 3.2 Sources of the 2,760 training rows

| Source | Rows | Notes |
|---|---|---|
| Rows derived from `sujalgawas/aya-json-sft-dataset` (revision `e26e691b`, train split) | 1,433 | Six document templates. 1,411 rows are augmented by our code: dates reworded while the answer stays `YYYY-MM-DD`, amounts with separators, enum labels reworded, and one to three fields removed from the text with the answer set to `null`. |
| Schema-first rows from the structure of real schemas in `epfl-dlab/JSONSchemaBench` (train splits) | 1,176 | Field names, nesting, enums and `$defs` come from real schemas. Documents and values come from our own code. No LLM is used. |
| Rows converted from `paraloq/json_data_extraction` | 151 | The licence is unclear (see section 9). |

Statistics of the final set: 16.9% of answer values are `null`, only 15% of ISO dates appear verbatim in the document, and 337 rows use `$defs` or `$ref`.

### 3.3 CuratorKIT pass
2,994 rows were pooled. CuratorKIT removed 140 near-duplicates (exact and MinHash), and its secrets gate rejected 22 rows. These are false positives on synthetic ids, and we kept the gate on. That left 2,832 rows. A length cap and our split guard left 2,760. One more row was over 2,560 tokens, so training used 2,759.

CuratorKIT's export formats drop the system prompt and the document, so our scripts write the final training files from the ids that survive its gates.

### 3.4 Splits and a leakage mistake that we caught
Splits are assigned at the schema level, before generation. `val` has new documents with seen schemas. `test` has new schemas in seen domains. `test_b` is a whole new domain (FieldBench). `test_c` is 143 real receipts.

Midway, the upstream dataset changed: old training rows moved into `val` and `test`. Our first SFT v2 build then had leakage: 88 of 120 `val` rows and 84 of 120 `test` rows shared field-name sets with training. A verification script found this because ids were missing. We rebuilt from the current dataset, pinned its revision, and added a guard that drops any training row whose schema or field-name set appears in an evaluation set. The guard is part of the build.

One caveat remains: about 78 to 81% of the answer strings in `val` and `test` also occur in the training answers. These two splits flatter the model. Judge generalization on `test_b` and `test_c`.

## 4. Training

QLoRA through AlignTune with the TRL backend: 4-bit NF4 base, LoRA rank 32, alpha 64, dropout 0.05, all linear layers (60,456,960 trainable parameters), learning rate 1e-4 with cosine decay, 2 epochs, batch 1 with gradient accumulation 16, maximum length 2,560, seed 42. Training ran in fp16 on one free Kaggle T4: 346 optimizer steps in 4 hours 39 minutes. The loss covers the JSON answer only (about 14% of the tokens).

Problems that we found and fixed (each one would have silently damaged the result):

- **Emulated bf16.** A T4 reports that it supports bf16, but it emulates it, about four times slower. We use bf16 only on GPUs with compute capability 8 or higher.
- **fp16 crash.** The trainer converts trainable weights to bf16 inside its constructor, and fp16 training then crashes. A small trainer subclass puts the weights back to fp32 when the optimizer is created.
- **No chat template.** AlignTune with `task_type="instruction_following"` turns chat rows into plain strings. We use `task_type="sft"`.
- **Loss on every token.** The Cohere template has no markers for the answer. We patched it with `{% generation %}` markers. The rendered text is identical, and the loss now covers only the answer and its two end tokens.

## 5. Evaluation

AuditKIT runs every evaluation. We add our own metrics as AuditKIT metrics, and its built-in metrics appear with the prefix `ak_`. Greedy decoding, 1,280 new tokens. (A 384-token limit cut long answers. Limits of 1,280, 1,536 and 2,048 gave identical outputs for the base model and the GRPO adapter.)

- **field_f1:** a score for each field value against the gold answer. Correct nulls count.
- **doc_exact:** every field in the document is correct. This is the strictest metric.
- **null_acc:** missing values are answered with `null`.
- **hallucination_rate:** values that the document does not support. We report a raw and an adjusted rate.
- **parse_ok, schema_valid:** the output is valid JSON, and follows the schema. These are **not** our headline, because constrained decoding makes them near 100% for any model.

We test differences on the same documents (paired). A paired bootstrap gives a 95% interval, and an exact McNemar test covers doc_exact.

All metric definitions are in one file (`eval.py`) and are used by every script. `verify_results.py` recomputes every number from the raw outputs, and checks the document counts, the model names, and that no two systems have identical outputs. We only quote numbers from runs that it marks VERIFIED.

## 6. Results

### 6.1 Main result: the SFT v2 adapter against the base model
Kaggle T4, fp16, all rows from one run.

| split | system | parse_ok | field_f1 | doc_exact | null_acc | hallucination (adjusted) |
|---|---|---|---|---|---|---|
| val (n=120) | base | 1.000 | 0.971 | 0.608 | 0.955 | 0.004 |
| val | SFT v2 | 1.000 | 0.998 | 0.975 | 1.000 | 0.000 |
| test (n=120) | base | 1.000 | 0.962 | 0.533 | 0.939 | 0.004 |
| test | SFT v2 | 1.000 | 0.997 | 0.958 | 1.000 | 0.000 |
| test_b (n=415) | base | 1.000 | 0.795 | 0.222 | 0.838 | 0.049 |
| test_b | SFT v2 | 0.983 | 0.815 | 0.263 | 0.873 | 0.045 |
| test_c (n=143) | base | 0.993 | 0.447 | 0.000 | 0.920 | 0.041 |
| test_c | SFT v2 | 0.993 | 0.494 | 0.000 | 0.962 | 0.027 |

Paired differences (SFT v2 minus base):

- `val` and `test`: doc_exact +0.367 and +0.425, significant.
- `test_b`: field_f1 +0.020 [+0.002, +0.038] and null_acc +0.036 are significant. **doc_exact +0.041 [-0.002, +0.084] is not significant** (McNemar p = 0.075). parse_ok fell from 1.000 to 0.983, a significant drop caused by broken JSON on some documents.
- `test_c`: field_f1 +0.047 [+0.022, +0.073] is significant. **No system gets a single receipt fully correct.**

In plain words: SFT v2 is far better on documents like its training data and somewhat better on a new domain and on real receipts. It is not a solved problem on real documents.

### 6.2 The earlier adapter memorized the templates
A teammate's run on another GPU (own adapter, different machine, so compare only inside this table):

| split | base doc_exact | early SFT doc_exact | SFT v2 doc_exact |
|---|---|---|---|
| val | 0.550 | 0.983 | 0.975 |
| test | 0.517 | 0.983 | 0.975 |
| test_b | 0.210 | 0.010 | 0.246 |

The early SFT adapter (trained on the first dataset) scores 0.983 on documents that look like its training data and collapses to 0.010 on a new domain (parse_ok 0.839, field_f1 0.520). More varied data (augmented dates, amounts and enums, removed fields, real schema structures) removes the collapse. Date normalization is a clear example: on `test_b` the base model leaves dates unnormalized 83.6 times per 100 documents (error count from our taxonomy tool), and SFT v2 0.0 times.

### 6.3 Negative result 1: the GRPO adapter
An adapter from an earlier GRPO attempt matched the base model within about one point on every metric and split (for example `val` doc_exact 0.340 for both, `test_b` doc_exact 0.227 against 0.224). That run used the first dataset, so the numbers are not comparable with the tables above. We did not investigate why GRPO changed nothing.

### 6.4 Negative result 2: DPO on top of SFT v2
**Why we tried it.** The error counts of SFT v2 showed new faults: values turned into `null` (58 and 108 per 100 documents on `test_b` and `test_c`, against 1.2 and 0.7 for the base model), extra keys, invented values, wrong types and some malformed JSON.

**Data.** 1,415 preference pairs (chosen = the gold JSON, rejected = a worse answer), 1,410 used. 247 rejected answers are real mistakes of SFT v2, sampled on 2,081 new prompts that were guarded against all training and evaluation sets. Only 277 of those 2,081 answers were wrong, because the prompts are easy. The remaining 1,168 rejected answers come from rules (for example a value turned into `null`, an extra key, a wrong type), with the mix set from the error counts. Both directions of the null error are included, so the model cannot learn "write more nulls".

**Training.** AlignTune, LoRA rank 16, beta 0.1, learning rate 1e-5 (untuned), 1 epoch, 89 steps, bf16. The first-step loss was 0.6931 (= ln 2), so the policy started equal to the reference. The loss fell from about 0.68 to 0.48, with no NaN. These are training numbers.

**Success rule, fixed before any result:** on `test_b`, doc_exact and field_f1 must improve with intervals above 0; on `test_c`, field_f1 must improve and doc_exact must not fall; on `val` and `test`, no significant drop.

**Result: the rule fails.** (One teammate GPU, greedy, 1,280 tokens, one precision for all rows.)

| `test_b` | parse_ok | field_f1 | doc_exact |
|---|---|---|---|
| SFT v2 adapter on the 4-bit base | 0.976 | 0.804 | 0.258 |
| merged SFT v2, no DPO (control) | 0.954 | 0.777 | 0.222 |
| merged SFT v2 plus DPO | 0.954 | 0.774 | 0.227 |

- DPO against the SFT v2 adapter looks worse (doc_exact -0.031, field_f1 -0.029, significant).
- But DPO against the merged model that it started from is a tie: doc_exact +0.005 [-0.010, +0.019] (p = 0.75), field_f1 -0.002 [-0.015, +0.011]. On `test_c`, field_f1 is 0.491 for both. DPO changed 23% of the `test_b` answers with no net gain or loss.
- The drop comes from **merging**: the merged model is worse than the adapter on the 4-bit base by 0.036 doc_exact [-0.058, -0.017] (p = 0.0007) on `test_b`. The merged weights are quantized to 4 bit again when loaded, and that is not bit-identical.
- `val` and `test`: no significant change.

The control row is what makes this readable. Without it, the drop against the adapter would look like a DPO failure.

**Decision:** the submitted model is the SFT v2 adapter. The DPO adapter is published as a documented negative result.

**Caveats.** The SFT v2 adapter rows and the merged rows were replayed from AuditKIT's cache of an earlier run on the same machine, not recomputed. Their outputs agree with the independent Kaggle run on 120 of 120 `val` and `test` answers and on 90% of `test_b` answers. The DPO rows were computed fresh. This is one run, one seed, an untuned learning rate and few, easy on-policy pairs. It shows that this recipe did not help, not that DPO cannot.

## 7. Cost and speed (what we measured)

- **Training:** SFT v2 took 4 hours 39 minutes on one free Kaggle T4. The DPO training time was not reported.
- **Inference speed, Kaggle T4:** the adapter processes about 0.04 documents per second (about 22,000 to 25,000 seconds per 1,000 documents). The base model is faster on this setup: 0.13 to 0.30 documents per second. We did not investigate why the adapter is slower.
- **Teammate's GPU:** about 0.08 documents per second for the DPO model (model of the GPU not reported).
- **Not done:** we did **not** run a closed-model baseline, and we did **not** compute a cost per 1,000 documents against closed-API pricing. We have no verified price source or closed-model output, so we do not quote a number.

## 8. What worked, what did not

**Worked**
- Schema-first gold plus augmentation (dates, amounts, enums, removed fields) removed the template memorization of the early adapter.
- A split guard and a pinned dataset revision caught and fixed a real leakage problem.
- Recomputing every metric from raw outputs, and checking that the model name includes the token limit and the precision, caught cached and identical-output bugs early.
- A fixed success rule and a control row prevented a wrong story about DPO.

**Did not work**
- GRPO (earlier attempt): no change.
- DPO: no measurable effect.
- Merging the adapter into the weights: a significant loss on the new domain.

## 9. Limitations and open items

- **Real documents are not solved.** `test_c` has no fully correct receipt for any system, and field_f1 is below 0.5. Typical faults: values turned into `null`, and extra keys. A few receipts produce very many extra keys.
- **New domain.** The doc_exact gain on `test_b` is not significant, and parse_ok fell from 1.000 to 0.983.
- **One run, one seed, one learning rate.** We have no variance estimate across seeds.
- **Missing from the evaluation plan:** base and SFT with constrained decoding, and a closed-model reference with a cost comparison. We did not run them.
- **Synthetic, template-like training documents.** OCR noise, long text and tables are much harder.
- **Hallucination rate.** The raw rate on `test_b` is higher than for the base model (0.253 against 0.201), while the adjusted rate is not different. We did not investigate this.
- **Speed.** The adapter is slow on a T4.
- **Licence.** The base model is CC-BY-NC-4.0, so the adapters use the same non-commercial licence. 151 of the 2,760 training rows come from `paraloq/json_data_extraction`, whose Hub tag says Apache-2.0 while its card says "intended for research purposes only". The team has not resolved this. The schemas from JSONSchemaBench keep their upstream licences. Some rows of the upstream dataset were written with Qwen2.5-7B-Instruct as teacher.
- **Data is not published.** The curated dataset and the raw generation logs are not yet on Hugging Face. They stay local until the licence question is closed.

## 10. Reproduce it

| Step | Command or file |
|---|---|
| Build the training data | `data/build_sft_v2.py` (reads the pinned dataset, runs CuratorKIT, applies the split guard, writes `STATS.md`) |
| Train SFT v2 | `python train_sft.py` (`--dry-run` first; `smoke_test_sft.sh` runs a short test) |
| Evaluate | `python eval.py --systems base sft_v2=<adapter> --splits val test test_b test_c --data-dir <folder>` |
| Verify | `python verify_results.py <results folder> --data-dir <folder>` |
| Compare | `python compare_results.py <results folder> base sft_v2` |
| DPO | `data/build_dpo_pairs.py`, `merge_sft_adapter.py`, `train_dpo.py`, `smoke_test_dpo.sh` (first-step loss must be about 0.693) |

Versions: AlignTune 0.1.33 (TRL 1.7.1, PEFT 0.21.2, transformers 5.20.2), AuditKIT 1.1.6, CuratorKIT 0.1.24. Seed 42. The plan and progress log of the DPO stage are in `docs/DPO_PLAN.md`, and a beginner walkthrough is in `docs/PROJECT_EXPLAINER.html`.

## 11. Artifacts

- Adapter (the submitted model): https://huggingface.co/monson2002/tiny-aya-sft-v2-adapter
- DPO adapter (negative result): https://huggingface.co/monson2002/tiny-aya-sft-v2-dpo-adapter
- Code and results: https://github.com/Monson2002/lexsi-hackathon
- Base model: https://huggingface.co/CohereLabs/tiny-aya-global

Data statement: all data is synthetic or public. No real personal data was used. The `test_c` receipts are real public documents from the FieldBench set.
