# What we learned teaching a 3B model to follow any JSON Schema (and why DPO did nothing)

> Fine-tuning a small Aya model (Tiny Aya Global, 3.35B) for **schema-conditioned structured extraction**: the input is a document plus a JSON Schema, the output is one JSON object that follows the schema, with `null` for values that are not in the document. Lexsi Labs hackathon, Track 1 (fine-tuning and data curation). Toolkits chained: CuratorKIT (data), AlignTune (training), AuditKIT (evaluation).
>
> **Links:** [Demo](https://ghostinthegradient.vercel.app/) (runs on a T4 that sleeps after 15 minutes, so the first request can be slow) | [Adapter (submitted model)](https://huggingface.co/monson2002/tiny-aya-sft-v2-adapter) | [DPO adapter (negative result)](https://huggingface.co/monson2002/tiny-aya-sft-v2-dpo-adapter) | [Abstract](docs/ABSTRACT.md) | [Technical write-up](docs/WRITEUP.md) | [Beginner walkthrough](docs/PROJECT_EXPLAINER.html) | [DPO plan and log](docs/DPO_PLAN.md)

*Lexsi Labs hackathon, Track 1: fine-tuning and data curation. We fine-tuned Tiny Aya Global with CuratorKIT, AlignTune and AuditKIT. This write-up is the honest version: what worked, what did not, and what we did not measure.*

## TL;DR

- **Task:** give the model a document and a JSON Schema. It must return one JSON object that follows the schema, with values taken from the document and `null` for anything missing.
- **What worked:** a QLoRA adapter trained on 2,760 carefully built examples. It raises document exact match from 0.608 to 0.975 on in-domain validation and from 0.533 to 0.958 on in-domain test documents.
- **What is still hard:** a new domain and real receipts. On a new domain, document exact match goes from 0.222 to 0.263, and that gain is **not** statistically significant. On 143 real receipts, no system, ours or the base model, gets a single receipt fully correct.
- **What did not work:** a GRPO adapter (no change), DPO on top of our adapter (no measurable effect), and merging the adapter into the base weights (a significant loss, about 3.6 points of document exact match on the new domain).
- **Everything is open:** adapters, code, verified result tables and a demo. Links are at the end.

## 1. The task, and why "any schema" is the hard part

Structured extraction sounds easy until you ask for the general skill. A model that fills in one fixed form is not useful. We want a model that follows whatever schema it is given: nested objects, arrays, enums, optional fields, and fields that look related but are absent from the document.

So the model must follow four rules:

1. The output is valid JSON and follows the schema.
2. Every value comes from the document. The model never invents a value.
3. A missing value is `null`.
4. The model changes a value (for example a date) only when the schema asks for it.

We test three things the model never saw in training: new documents with known schemas (`val`), new schemas (`test`), and a whole new domain (FieldBench `test_b`). A small set of real receipts (`test_c`) checks real-world text.

## 2. Our first lesson: 0.983 can mean nothing

Our first adapter scored **0.983** document exact match on documents that looked like its training data. On a new domain it scored **0.010**. It had memorized six document templates.

Any score on data that resembles the training set flatters the model. In our case, about 78 to 81 percent of the answer strings in the in-domain test sets also appear in the training answers. We now treat the new-domain and real-document numbers as the honest ones, and we report the in-domain numbers with a warning.

## 3. Data: build the answer first

We do not use "a strong model extracts the answer" as gold. That makes the gold as good as the extractor, with its mistakes baked in.

Instead, we build **schema first**: we pick a schema, sample a record that satisfies it, and then write the document from the record. The gold answer is then exact by construction.

Our 2,760 training rows come from three sources:

| Source | Rows | Notes |
|---|---|---|
| A public dataset (`sujalgawas/aya-json-sft-dataset`) | 1,433 | Six templates. Our code rewords dates, adds amount separators, rewords enum labels, and removes one to three fields from the text so the answer is `null`. |
| Structure of real schemas (`epfl-dlab/JSONSchemaBench`) | 1,176 | Field names, nesting and enums come from real schemas. Our own code generates the values and documents. |
| `paraloq/json_data_extraction` | 151 | The licence is unclear (see the limits). |

CuratorKIT handled the curation: gates for schema and length, exact and near-duplicate removal, and a secrets check. After the gates and a length cap, 2,760 rows remained.

Some features of the final set matter more than its size: 16.9 percent of answer values are `null`, and only 15 percent of ISO dates appear verbatim in the document. The model has to convert the rest. That is where a lot of the early failures came from: the base model left dates unnormalized 83.6 times per 100 documents on the new domain, and the trained adapter did so 0.0 times.

### The leakage we almost shipped
Midway through the project the upstream dataset changed, and old training rows moved into the validation and test sets. Our first rebuild trained on schemas that were now in the evaluation sets: **88 of 120** validation rows and **84 of 120** test rows shared field-name sets with training.

A verification script caught it, because ids were missing. We rebuilt from a pinned revision and added a split guard that drops any training row whose schema or field-name set appears in an evaluation set. The guard is now part of the build.

## 4. Training on a free T4

We trained a QLoRA adapter with AlignTune: a 4-bit base, LoRA rank 32 on all linear layers (60 million trainable parameters), 2 epochs, loss on the JSON answer only. It took 4 hours and 39 minutes on one free Kaggle T4.

Four details cost us time, so we list them:

- A T4 reports that it supports bf16, but it emulates it, about four times slower. Use bf16 only on GPUs with compute capability 8 or higher.
- The trainer casts trainable weights to bf16 in its constructor, which crashes fp16 training. We restore fp32 when the optimizer is created.
- With the wrong task type, AlignTune turned our chat rows into plain strings and the chat template was never applied.
- The chat template had no markers for the answer, so the loss covered every token. We patched the template so the loss covers only the answer and its end tokens.

## 5. How we measure, and why validity is not the headline

Constrained decoding forces any model to write valid JSON, so schema validity is near 100 percent for every model. It does not show that the values are right. Our headline metrics are:

- **field F1:** a score for each field value. Correct nulls count.
- **document exact match:** every field in the document is correct.
- **null accuracy**, and **hallucination rate** (values that the document does not support).

Three habits kept us honest:

1. **Paired tests.** The same documents go to both systems. A bootstrap interval and an exact McNemar test tell us whether a difference is more than luck.
2. **Recompute from raw outputs.** A script recomputes every metric from the raw model outputs, and checks document counts and that no two systems have identical outputs. It caught a cached-results bug that gave two different models the same score.
3. **Control rows.** More on this below. It changed our conclusion about DPO.

## 6. Results

All rows are from one Kaggle T4 run, greedy decoding, 1,280 new tokens.

| split | system | parse ok | field F1 | doc exact | null acc |
|---|---|---|---|---|---|
| `val` (120) | base | 1.000 | 0.971 | 0.608 | 0.955 |
| `val` | **adapter** | 1.000 | 0.998 | **0.975** | 1.000 |
| `test` (120) | base | 1.000 | 0.962 | 0.533 | 0.939 |
| `test` | **adapter** | 1.000 | 0.997 | **0.958** | 1.000 |
| `test_b`, new domain (415) | base | 1.000 | 0.795 | 0.222 | 0.838 |
| `test_b` | **adapter** | 0.983 | 0.815 | 0.263 | 0.873 |
| `test_c`, real receipts (143) | base | 0.993 | 0.447 | 0.000 | 0.920 |
| `test_c` | **adapter** | 0.993 | 0.494 | 0.000 | 0.962 |

What the paired tests say:

- In domain, the gains are large and significant (+0.367 and +0.425 document exact match).
- On the new domain, field F1 (+0.020) and null accuracy (+0.036) improve significantly. **Document exact match (+0.041, interval -0.002 to +0.084) does not.** The parse rate also fell from 1.000 to 0.983, because of broken JSON on a few documents.
- On real receipts, field F1 improves (+0.047, significant), but **document exact match is 0.000 for every system**.

The adapter fixed the failures that were easy to fix: date normalization, markdown fences around the JSON, and missing-field handling. It did not make the model good at messy, real documents. We think that is the honest summary.

## 7. Three things that did not work

### GRPO
An earlier GRPO adapter matched the base model within about one point on every metric and split. We did not find out why.

### DPO, and the control that saved us
SFT left new faults. The model now turns real values into `null` (58 and 108 times per 100 documents on the new domain and on receipts, against 1.2 and 0.7 for the base model), adds extra keys, invents values, and sometimes writes broken JSON.

So we tried DPO on top of the adapter. We built 1,415 preference pairs: the gold JSON as the good answer and a worse answer as the bad one. 247 of the bad answers were the model's own mistakes, sampled on 2,081 new prompts. The rest came from rules that corrupt the gold: a value turned into `null`, an extra key, a wrong type, truncated JSON. We included both directions of the null error, so the model could not learn to "write more nulls".

We **fixed the success rule before we saw any result**. Training looked healthy: the first-step loss was 0.6931 (that is ln 2, so the policy started equal to its reference), the loss fell from about 0.68 to 0.48, and there was no NaN.

Then we evaluated on a teammate's GPU, and DPO looked **worse** than our adapter on the new domain: document exact match 0.227 against 0.258. That reads like a failure of DPO. But the evaluation had a control row: the merged model, which is what DPO actually started from, without DPO.

| `test_b`, new domain | parse ok | field F1 | doc exact |
|---|---|---|---|
| adapter on the 4-bit base | 0.976 | 0.804 | 0.258 |
| merged model, no DPO (control) | 0.954 | 0.777 | 0.222 |
| merged model plus DPO | 0.954 | 0.774 | 0.227 |

(This table comes from the teammate's GPU run, so its first row differs slightly from the T4 table above. Compare systems only inside one run.)

DPO against the control is a tie: document exact match +0.005 (interval -0.010 to +0.019, p = 0.75). DPO changed 23 percent of the answers, but with no net gain or loss. **DPO did nothing measurable.** The apparent drop came from somewhere else.

### Merging
That somewhere else is the merge. Merging the adapter into the weights, and loading the result in 4 bit again, is not bit-identical to the adapter on the 4-bit base. On the new domain it costs **0.036 document exact match** (interval -0.058 to -0.017, p = 0.0007) and 0.027 field F1. If you serve this model, keep it as an adapter on the 4-bit base. Do not merge it.

Without the control row, we would have published "DPO hurts". The truth is "DPO is neutral, and our merge step hurts". These are different lessons.

One caveat on the DPO test: it is one run, one seed, an untuned learning rate (1e-5), one epoch, and only 247 on-policy pairs on easy prompts (the model answered 87 percent of them fully correctly). It shows that **this recipe** did not help. It does not show that DPO cannot help.

## 8. Limits we want you to see

- **Real documents are not solved.** Document exact match on receipts is 0.000, and field F1 is below 0.5.
- **The new-domain gain is not significant** for document exact match, and the parse rate dropped.
- **One run, one seed.** We have no spread across seeds.
- **Not done:** base and adapter with constrained decoding, and a closed-model baseline with a cost comparison. We have no verified price source, so we do not quote a cost.
- **Slow on a T4:** the adapter handles about 0.04 documents per second, against 0.13 to 0.30 for the base model on the same setup. We did not investigate why.
- **Licence:** the base model is CC-BY-NC-4.0, so the adapters are non-commercial. 151 of the 2,760 training rows come from a dataset whose licence note conflicts (the Hub tag says Apache-2.0, the card says "intended for research purposes only"). We have not resolved this, so we have not published our curated dataset.
- Training documents are synthetic and template-like.

## 9. Try it, check it, reuse it

- **Adapter (the model we submit):** [`monson2002/tiny-aya-sft-v2-adapter`](https://huggingface.co/monson2002/tiny-aya-sft-v2-adapter)
- **DPO adapter (a documented negative result):** [`monson2002/tiny-aya-sft-v2-dpo-adapter`](https://huggingface.co/monson2002/tiny-aya-sft-v2-dpo-adapter)
- **Code, verified result tables and the full write-up:** [github.com/Monson2002/lexsi-hackathon](https://github.com/Monson2002/lexsi-hackathon)
- **Demo:** [ghostinthegradient.vercel.app](https://ghostinthegradient.vercel.app/) (the page states a T4 GPU that sleeps after 15 minutes, so the first request can be slow)
- **Base model:** [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global)

To check our numbers, run `verify_results.py` on any result folder. It recomputes every metric from the raw outputs. We only quote numbers from runs that it marks VERIFIED.

*Toolkits: CuratorKIT 0.1.24 (data), AlignTune 0.1.33 (training), AuditKIT 1.1.6 (evaluation). Seed 42.*

---

## Repository guide

| Path | What it is |
|---|---|
| `eval.py` | AuditKIT evaluation harness and the single definition of every metric |
| `verify_results.py` | recomputes every metric from raw outputs; prints VERIFIED or NOT VERIFIED |
| `compare_results.py`, `error_taxonomy.py` | paired comparison with the DPO success rule; failure-type counts |
| `train_sft.py`, `smoke_test_sft.sh` | QLoRA SFT through AlignTune, and a short smoke test |
| `train_dpo.py`, `merge_sft_adapter.py`, `smoke_test_dpo.sh` | the DPO stage (merge SFT v2 into the weights, train, smoke test) |
| `fieldbench_to_jsonl.py` | converts FieldBench into the `test_b` and `test_c` evaluation files |
| `data/` | data builders: augmentation, schema-first generator, CuratorKIT build, leakage check, DPO pair builder, on-policy sampler. Generated data is not tracked (see the licence note in the limits) |
| `results/` | verified result tables, logs and paired comparisons |
| `sft_v2_adapter/` | model card, config and training log (the weights are on Hugging Face) |
| `docs/` | abstract, write-up, DPO plan and log, CuratorKIT usage notes, beginner walkthrough |
| `AGENTS.md` | rules for contributors and coding agents |

## Reproduce

| Step | Command or file |
|---|---|
| Build the training data | `data/build_sft_v2.py` (reads the pinned dataset, runs CuratorKIT, applies the split guard, writes `STATS.md`) |
| Train SFT v2 | `python train_sft.py` (`--dry-run` first; `smoke_test_sft.sh` runs a short test) |
| Evaluate | `python eval.py --systems base sft_v2=<adapter> --splits val test test_b test_c --data-dir <folder>` |
| Verify | `python verify_results.py <results folder> --data-dir <folder>` |
| Compare | `python compare_results.py <results folder> base sft_v2` |
| DPO | `data/build_dpo_pairs.py`, `merge_sft_adapter.py`, `train_dpo.py`, `smoke_test_dpo.sh` (the first-step loss must be about 0.693) |

Versions: AlignTune 0.1.33 (TRL 1.7.1, PEFT 0.21.2, transformers 5.20.2), AuditKIT 1.1.6, CuratorKIT 0.1.24. Seed 42. The full technical write-up is in [`docs/WRITEUP.md`](docs/WRITEUP.md).

---

## Appendix: all results by run, with dates (status on Nov 15, 2026)

Every number below comes from a result file in this repo (`results/`) or in the model repo [`monson2002/tiny-aya-sft-v2-adapter`](https://huggingface.co/monson2002/tiny-aya-sft-v2-adapter) under `results/`. Nothing is estimated. DPO was trained and evaluated: it gave **no gain** over SFT v2 (see the DPO section).

### How to read the numbers (for beginners)
- **Splits.** `val` and `test` (also called `test_a` in the oldest run) hold documents like the training ones, so scores there are high and flatter the model. `test_b` is a whole new domain (FieldBench). `test_c` is 143 real receipts. The honest test of "did it learn the general skill" is `test_b` and `test_c`.
- **doc_exact:** the share of documents where every field is right. This is the strictest number. **field_f1:** a score per field value, so partly right counts. **parse_ok:** the answer is valid JSON. **schema_valid:** valid JSON that follows the schema. **null_acc:** missing values were answered with `null`. **hallucination_rate:** values that the document does not support.
- **Schema validity is not our headline.** A decoding trick (constrained decoding) makes any model write valid JSON, so validity says little about correct values.
- **Same run only.** Compare systems only inside one run. The base model scores a little differently on different machines (see runs C and D).

### Runs, in date order
| Run | Files dated | Systems | Data revision | Machine |
|---|---|---|---|---|
| A. Token-cap test | Nov 10 to 11, 2026 | base, GRPO | first dataset (has a `test_a` split) | Kaggle |
| B. SFT v2 evaluation by a teammate | Nov 12, 2026 | base, early SFT, SFT v2 (teammate's own adapter) | `e26e691b` | teammate's GPU |
| C. HF adapter evaluation | Nov 13, 2026 | base, SFT v2 (HF adapter) | `e26e691b` | Kaggle T4 |

**Run A is not comparable with B and C**: the dataset changed between them (rows moved between train, val and test), so the base model scores differently (`val` doc_exact 0.340 in A against 0.550 and 0.608 later).

### Run A: base against GRPO adapter (first dataset, Nov 10 to 11)
| split | system | field_f1 | doc_exact | null_acc | hallucination_rate |
|---|---|---|---|---|---|
| val | base | 0.927 | 0.340 | 0.804 | 0.044 |
| val | GRPO | 0.927 | 0.340 | 0.806 | 0.044 |
| test_a | base | 0.907 | 0.400 | 0.741 | 0.035 |
| test_a | GRPO | 0.905 | 0.390 | 0.738 | 0.036 |
| test_b | base | 0.794 | 0.227 | 0.844 | 0.200 |
| test_b | GRPO | 0.796 | 0.224 | 0.838 | 0.203 |
| test_c | base | 0.446 | 0.000 | 0.919 | 0.214 |
| test_c | GRPO | 0.447 | 0.000 | 0.920 | 0.211 |

The same files were produced with caps of 1280, 1536 and 2048 new tokens and were identical, so we use 1280 (a 384-token cap cut long answers).
**In plain words:** the GRPO adapter behaves like the base model: every difference is about one point or less. It did not teach the model anything measurable.

### Run B: base, early SFT, SFT v2 (teammate's adapter, Nov 12)
| split | system | parse_ok | field_f1 | doc_exact |
|---|---|---|---|---|
| val | base | 1.000 | 0.966 | 0.550 |
| val | early SFT | 1.000 | 0.998 | 0.983 |
| val | SFT v2 | 1.000 | 0.998 | 0.975 |
| test | base | 1.000 | 0.962 | 0.517 |
| test | early SFT | 1.000 | 0.999 | 0.983 |
| test | SFT v2 | 1.000 | 0.998 | 0.975 |
| test_b | base | 1.000 | 0.794 | 0.210 |
| test_b | early SFT | 0.839 | 0.520 | 0.010 |
| test_b | SFT v2 | 0.957 | 0.783 | 0.246 |
| test_c | base | 0.993 | 0.448 | 0.000 |
| test_c | early SFT | 0.699 | 0.313 | 0.000 |
| test_c | SFT v2 | 0.986 | 0.484 | 0.000 |

**In plain words:** the early SFT adapter memorised the training templates. It scores 0.983 on documents that look like training data and collapses to 0.010 on a new domain. SFT v2 (more varied data) keeps the high in-domain score and does not collapse.

### Run C: base against the public HF SFT v2 adapter (Nov 13)
This is the adapter that is public and that the DPO stage builds on.
| split | system | parse_ok | schema_valid | field_f1 | doc_exact | null_acc | hallucination_rate (adjusted) |
|---|---|---|---|---|---|---|---|
| val | base | 1.000 | 0.958 | 0.971 | 0.608 | 0.955 | 0.004 |
| val | SFT v2 (HF) | 1.000 | 1.000 | 0.998 | 0.975 | 1.000 | 0.000 |
| test | base | 1.000 | 0.925 | 0.962 | 0.533 | 0.939 | 0.004 |
| test | SFT v2 (HF) | 1.000 | 1.000 | 0.997 | 0.958 | 1.000 | 0.000 |
| test_b | base | 1.000 | 0.827 | 0.795 | 0.222 | 0.838 | 0.049 |
| test_b | SFT v2 (HF) | 0.983 | 0.875 | 0.815 | 0.263 | 0.873 | 0.045 |
| test_c | base | 0.993 | 0.944 | 0.447 | 0.000 | 0.920 | 0.041 |
| test_c | SFT v2 (HF) | 0.993 | 0.965 | 0.494 | 0.000 | 0.962 | 0.027 |

Paired differences against base (same documents, bootstrap 95% interval):
- `val` and `test`: doc_exact +0.367 and +0.425, significant.
- `test_b`: field_f1 +0.020 [+0.002, +0.038] and null_acc +0.036 are significant. doc_exact +0.041 [-0.002, +0.084] is **not** significant (McNemar p = 0.075). parse_ok fell from 1.000 to 0.983, a significant drop.
- `test_c`: field_f1 +0.047 [+0.022, +0.073] is significant. No system gets a single receipt fully correct.

**In plain words:** SFT v2 is much better on documents like its training data and somewhat better on new domains and real receipts. It is not a solved problem on real documents.

### DPO stage (Nov 15, 2026): trained and evaluated, no gain
A second training stage, DPO, was added on top of SFT v2 to fix its remaining faults (values turned into `null`, extra keys, invented values, wrong types, malformed JSON). Plan, success rule (fixed before any result) and progress log: [`docs/DPO_PLAN.md`](docs/DPO_PLAN.md).
- **Preference data:** 1,415 pairs (chosen = the gold JSON, rejected = a worse answer), 1,410 used. 247 rejected answers are real mistakes of the SFT v2 model (sampled on 2,081 new prompts; only 277 of its answers were wrong, because those prompts are easy); the rest are made by code, with the mix set from the error counts of the SFT v2 adapter. Statistics: `data/dpo/STATS_dpo.md`.
- **Training (verified from the run files):** LoRA r16 on the merged SFT v2 model, beta 0.1, lr 1e-5, 1 epoch, 89 steps. First-step loss 0.6931 (= ln 2, policy equal to reference), no NaN, loss 0.68 to 0.48, training pair accuracy about 0.89. Adapter: [`monson2002/tiny-aya-sft-v2-dpo-adapter`](https://huggingface.co/monson2002/tiny-aya-sft-v2-dpo-adapter) (needs the merged SFT v2 weights).
- **Evaluation** (teammate's GPU, greedy, 1280 tokens, one machine and precision for all rows; all result files recomputed and VERIFIED here; files in `results/dpo_eval/`). `sft_v2_hf` = the SFT v2 adapter on the 4-bit base. "merged" = SFT v2 merged into the weights without DPO (the control). "DPO" = the DPO adapter on the merged model.

| split | system | parse_ok | schema_valid | field_f1 | doc_exact | null_acc |
|---|---|---|---|---|---|---|
| test_b | sft_v2_hf | 0.976 | 0.858 | 0.804 | 0.258 | 0.866 |
| test_b | merged (no DPO) | 0.954 | 0.843 | 0.777 | 0.222 | 0.845 |
| test_b | DPO | 0.954 | 0.851 | 0.774 | 0.227 | 0.838 |
| test_c | sft_v2_hf | 0.993 | 0.972 | 0.496 | 0.000 | 0.962 |
| test_c | merged (no DPO) | 0.993 | 0.965 | 0.491 | 0.000 | 0.956 |
| test_c | DPO | 1.000 | 0.951 | 0.491 | 0.000 | 0.958 |
| val | sft_v2_hf | 1.000 | 1.000 | 0.998 | 0.975 | 1.000 |
| val | DPO | 1.000 | 1.000 | 0.997 | 0.967 | 1.000 |
| test | sft_v2_hf | 1.000 | 1.000 | 0.997 | 0.958 | 1.000 |
| test | DPO | 1.000 | 1.000 | 0.998 | 0.967 | 1.000 |

- **The success rule fails.** On `test_b`, DPO against `sft_v2_hf` is *worse*: doc_exact -0.031 [-0.055, -0.007], field_f1 -0.029 [-0.045, -0.015], parse_ok -0.022 (all significant). On `test_c` field_f1 is -0.005 (not significant). On `val` and `test` there is no significant change.
- **DPO itself did nothing measurable.** Against the merged control (its real starting point), `test_b` doc_exact is +0.005 [-0.010, +0.019] (McNemar p = 0.75), field_f1 -0.002 [-0.015, +0.011]; `test_c` field_f1 is 0.000. DPO changed 23% of the `test_b` answers against the merged model, with no net gain or loss.
- **Merging costs accuracy.** The merged model is worse than the adapter on the 4-bit base: `test_b` doc_exact -0.036 [-0.058, -0.017] (McNemar p = 0.0007), field_f1 -0.027 [-0.043, -0.013]. This is why the DPO drop against `sft_v2_hf` is not a DPO effect. The merged weights are quantised to 4 bit again when loaded, which is not bit-identical to the adapter on the 4-bit base.
- **Decision:** the submitted model is the SFT v2 adapter (`monson2002/tiny-aya-sft-v2-adapter`). The DPO adapter is kept as a documented negative result.
- **Caveats:** the `sft_v2_hf` rows (all splits) and the merged control rows were replayed from AuditKIT's cache of an earlier run on the same machine (0 s in the log), not recomputed; their outputs agree with the independent Kaggle T4 run on 120 of 120 `val` and `test` answers and on 90% of `test_b` answers. The DPO rows were computed fresh. One run, one seed, one learning rate (1e-5, untuned), only 247 on-policy pairs on easy prompts: this shows that this DPO recipe did not help, not that DPO cannot help.

### What we know and do not know
- **Known:** SFT v2 fixed the main failure of the early SFT (collapse on a new domain). Date normalisation, markdown fences and missing-field handling improved a lot. The GRPO adapter did nothing measurable. DPO on top of SFT v2 did nothing measurable either, and merging SFT v2 into the weights costs about 3 points on `test_b`.
- **Not known:** why DPO did not help (untested guesses: too few and too easy on-policy pairs, an untuned learning rate, one epoch). Why the SFT v2 evaluation was slow (0.04 documents per second on the T4, and 12.9 seconds per document on the teammate's GPU).
- **Remaining weaknesses seen in the error counts of run B:** the model writes `null` where the document has a value (64 of 100 documents on `test_b`, against 1.4 for base), makes bracket errors in nested JSON, and puts a list into a string field.
- **Open licence question:** 151 of the 2,760 training rows come from `paraloq/json_data_extraction`, whose licence terms are unclear. The team has not decided yet.

### Rerun the checks
`python verify_results.py <results_dir>` recomputes every metric from the raw outputs and prints VERIFIED or NOT VERIFIED. `python compare_results.py <results_dir> base sft_v2_hf --no-rule` gives the paired comparison.
