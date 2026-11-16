# Teaching Tiny Aya Global to follow any JSON Schema

> Fine-tuning a small Aya model (Tiny Aya Global, 3.35B) for **schema-conditioned structured extraction**. Lexsi Labs hackathon, Track 1 (fine-tuning and data curation), toolkits CuratorKIT, AlignTune and AuditKIT.
>
> **Links:** [Demo](https://ghostinthegradient.vercel.app/) (runs on a T4 that sleeps after 15 minutes, so the first request can be slow) | [Adapter (submitted model)](https://huggingface.co/monson2002/tiny-aya-sft-v2-adapter) | [DPO adapter (negative result)](https://huggingface.co/monson2002/tiny-aya-sft-v2-dpo-adapter) | [Models collection](https://huggingface.co/collections/monson2002/lexsi-hackathon) | [Article version of this write-up](docs/HF_ARTICLE.md) | [Standalone abstract](docs/ABSTRACT.md) | [Beginner walkthrough](docs/PROJECT_EXPLAINER.html) | [DPO plan and log](docs/DPO_PLAN.md)

**One-line claim:** a small QLoRA adapter makes a 3.35B Aya model follow arbitrary JSON Schemas on in-domain documents (document exact match 0.533 → 0.958), with only a small, not statistically significant gain on a new domain, and we report three methods that did not help (GRPO, DPO, merging).

Every number in this document comes from a result file in this repository (`results/`) or in the Hugging Face repositories listed under Artifacts. Where we did not measure something, this document says so.

## Tracks and problem statements covered

- **Track 1** (fine-tuning and data curation), **problem statement 1**: structured output and extraction.

## Content warning

No harmful or NSFW text appears in this write-up. Training data is synthetic or public. The `test_c` evaluation set holds real public documents (mostly receipts), which contain business names, addresses and business phone numbers. We use them only for evaluation and do not redistribute the documents.

## Abstract

We fine-tune Tiny Aya Global (3.35B parameters) to read a document and a JSON Schema and return one JSON object, with `null` for any value that the document does not contain. We aim for the general skill of following arbitrary schemas, and we test it on schemas and a domain never seen in training.

**Method.** With CuratorKIT we built 2,760 training examples from three sources. The gold answer is exact by construction: we sample a record from the schema first and then write the document from it. Dates, amounts and enum labels appear in varied forms, and one to three fields are removed from the text, so the correct answer is `null`. A split guard removes any training schema that overlaps an evaluation set. With AlignTune we trained a QLoRA adapter on one free Kaggle T4 GPU, with the loss on the JSON answer only. With AuditKIT we evaluated on four splits: two in-domain splits, FieldBench `test_b` (a new domain, 415 synthetic documents) and `test_c` (143 real documents, mostly receipts). We report field-level F1, document exact match, null accuracy and hallucination rate. Schema validity is not a headline metric: constrained decoding makes it near 100% for any model.

**Results.** Against the base model, document exact match rises from 0.608 to 0.975 on `val` and from 0.533 to 0.958 on `test` (both significant). On the new domain the gain is small: field F1 rises from 0.795 to 0.815 (significant), but document exact match rises from 0.222 to 0.263 and the interval includes zero. On real documents, field F1 rises from 0.447 to 0.494 (significant), but no system gets a single document fully correct. An earlier adapter scored 0.983 in domain and 0.010 on the new domain, because it had memorized the templates. More varied data removed that collapse.

**Negative results.** An earlier GRPO adapter matched the base model within about one point on every split. A DPO stage on top of our adapter (1,415 preference pairs, 247 of them the model's own mistakes) did not help: against the merged model that it started from, the change was within noise. We also found that merging the adapter into the weights costs about 3 points of document exact match on the new domain.

**Limits.** All results come from one run and one seed. Training documents are synthetic and template-like. We did not evaluate constrained decoding, and we did not run a closed-model baseline or a cost comparison. The adapter is slow on a T4 (about 0.04 documents per second). Part of the training data has an unresolved licence note, and the model is non-commercial like its base.

## Headline before → after

| Metric | Base model | Our adapter | Significant? |
|---|---|---|---|
| Document exact match, in-domain `test` (120) | 0.533 | **0.958** | yes |
| Document exact match, new domain `test_b` (415, synthetic) | 0.222 | 0.263 | **no** (interval -0.002 to +0.084) |
| Field F1, real documents `test_c` (143) | 0.447 | 0.494 | yes |

**Base model + method in one line:** `CohereLabs/tiny-aya-global` (3.35B, 4-bit NF4) + a LoRA rank 32 adapter trained with SFT (QLoRA, AlignTune) on 2,760 curated schema-first examples (CuratorKIT), evaluated with AuditKIT.

## For judges: one minute

- **What we built:** an adapter that returns schema-valid JSON with correct values and correct nulls. In-domain document exact match went from 0.533 to 0.958.
- **What is not solved:** a new domain (gain not significant) and real documents (no document fully correct for any system, including the base model).
- **Negative results we publish:** GRPO (no change), DPO (no measurable effect, tested against a control), merging the adapter into the weights (a significant loss).
- **How to trust the numbers:** `verify_results.py` recomputes every metric from the raw outputs; we only quote runs it marks VERIFIED. Differences are tested on the same documents (paired bootstrap, McNemar).
- **Where to look:** results in "Results", negative results in "What did not work", limits in "Limitations". Code: the GitHub repo. Demo: link under Artifacts.

## Problem statement and approach

A user gives the model a **document** and a **JSON Schema**. The model must return one JSON object that:

- is valid JSON and follows the schema (types, enums, required keys, nesting),
- takes every value from the document and never invents a value,
- writes `null` for a value that the document does not contain,
- changes a value (for example a date) only when the schema asks for it.

The goal is the general skill of following arbitrary schemas. A model that learns one fixed form is not useful. So we test on schemas it never saw (`test`), on a domain it never saw (`test_b`), and on real documents (`test_c`). The prompt format is frozen and identical in training and evaluation: one system message, and one user message with a `<JSON_SCHEMA>` block and a `<DOCUMENT>` block.

```
CuratorKIT:  pool of examples  ->  gates (schema, length, duplicates, secrets)  ->  split guard  ->  2,760 rows
AlignTune:   QLoRA SFT on Tiny Aya Global (loss on the JSON answer only)         ->  SFT v2 adapter
AuditKIT:    base vs adapter on 4 splits, metrics recomputed from raw outputs    ->  verified tables
Extra:       DPO on top of SFT v2 (tested, no gain)    GRPO adapter (earlier, no change)
```

### Why this problem, and the real use case
Turning documents (invoices, support tickets, forms, contracts, clinical-style notes) into structured records is a common back-office job. A small open model that follows the caller's own schema is cheap to host and keeps documents on the user's own hardware. The hard parts are the ones we measure: correct nulls (no invented values), normalization (dates and amounts), and staying correct on documents unlike the training data.

### Research questions and hypotheses
1. Does schema-first, varied training data remove the template memorization that an earlier adapter showed? (Not pre-registered. Answer: yes, see Results.)
2. Does DPO on top of SFT fix the remaining faults? **Pre-registered**: the success rule was fixed before any DPO result existed. On `test_b`, document exact match and field F1 must improve with 95% intervals above 0. On `test_c`, field F1 must improve and document exact match must not fall. On `val` and `test`, no significant drop. (Answer: no, see Results.)

### Related work
- Tiny Aya: Bridging Scale and Multilingual Depth (arXiv 2603.11510), the base model family.
- JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models (arXiv 2501.10868), the source of the real schema structures that we reuse.
- SO-Bench: A Structural Output Evaluation of Multimodal LLMs (arXiv 2511.21750), related evaluation of structured output.
- AlignTune (arXiv 2602.09621) and CuratorKIT (arXiv 2606.21631), the hackathon toolkits.

## Hackathon repos (kits) used

We used three of the toolkits. We did not use AgentTune, SafeTune or CircuitKIT.

**CuratorKIT 0.1.24** (data curation): a schema gate and a length gate on every example, exact and MinHash near-duplicate removal, a secrets gate (detect-secrets), and provenance files. We did not use its `messages` and `dpo` exports, because they drop the system prompt and the document. Our scripts write the final files from the ids that survive its gates. The same pass curated the DPO pairs.

**AlignTune 0.1.33** (training): `create_sft_trainer` (task type `sft`, TRL backend) for QLoRA SFT, and `create_rl_trainer` (algorithm `dpo`, TRL backend) for DPO. It writes a provenance record (`lexsi_provenance.json`) for every run. We worked around three of its behaviors (see Method).

**AuditKIT 1.1.6** (evaluation): `ak.evaluate` with a `CallableModel` wrapping our generation, our own extraction metrics registered as AuditKIT metrics, and its built-in `JsonSchemaValid` and `FieldF1` metrics (shown with the prefix `ak_`). Its run cache caused one bug (see Method).

### How the repos connect
CuratorKIT produces the curated training file, AlignTune trains on it and writes the adapter, and AuditKIT evaluates the adapter. Our own code fills the gaps: the split guard and leak check, the extraction metrics (one shared definition in `eval.py`), the DPO pair builder, and `verify_results.py`, which recomputes every metric from AuditKIT's raw outputs. The diagram is under "Problem statement and approach".

## Models and licenses

- **Base Aya model:** `CohereLabs/tiny-aya-global`, 3.35B parameters, gated, **CC-BY-NC-4.0 (non-commercial)**. Loaded in 4 bit (NF4, double quantization), fp16 compute on a T4 (bf16 on Ampere or newer). Our adapters use the same licence.
- **Teacher model:** none for our own rows. The schema-first rows use our own generator code (no LLM). Some rows of the upstream dataset `sujalgawas/aya-json-sft-dataset` were written by its author with Qwen2.5-7B-Instruct as teacher, so those rows are distilled.
- **Closed-model baseline:** none. Not done.

## Datasets, benchmarks and licenses

### Sources of the 2,760 training rows

| Source | Rows | Licence |
|---|---|---|
| `sujalgawas/aya-json-sft-dataset` (revision `e26e691b`, train split), augmented by our code | 1,433 | Apache-2.0 (dataset card). Six document templates: invoices, tickets, contracts, clinical notes, shipping manifests, compliance audits. |
| Schema-first rows from the structure of real schemas in `epfl-dlab/JSONSchemaBench` (train splits) | 1,176 | The dataset card says MIT; the schemas inside keep their upstream licences. Documents and values come from our own code. |
| `paraloq/json_data_extraction`, converted | 151 | **Unresolved**: the Hub tag says Apache-2.0, the card also says "intended for research purposes only". |

Of the 1,433 upstream-derived rows, 1,411 are augmented: dates reworded while the answer stays `YYYY-MM-DD`, amounts with separators, enum labels reworded, and one to three fields removed from the text with the answer set to `null`. Statistics of the final set: 16.9% of answer values are `null`, only 15% of ISO dates appear verbatim in the document, and 337 rows use `$defs` or `$ref`.

### Languages covered
Effectively English only. We checked the 2,760 training documents: 1,217 contain any non-ASCII character, but 1,210 of those characters are em dashes, 72 are Cyrillic and 2 are CJK. We did **not** evaluate any other language, although the base model is multilingual.

### Train / test splits and leak check

| Split | Documents | What it tests |
|---|---|---|
| train | 2,760 (2,759 used) | |
| `val` | 120 | new documents, seen schemas |
| `test` | 120 | new schemas, seen domains |
| `test_b` | 415, synthetic (CC0: insurance, contracts, legal and SEC filings) | a whole new domain |
| `test_c` | 143, real public documents (131 receipts, 8 legal filings, 4 insurance claims) | real, messy text |

FieldBench's own datasheet warns that synthetic documents overestimate accuracy, so read `test_b` as a best case for a new domain, and never pool it with `test_c`.

**Leak check.** Splits are assigned at the schema level before generation. Our split guard drops any training row whose canonical schema (exact match) or field-name set also appears in `val`, `test`, `test_b` or `test_c`. It is part of the build, and its result is in `STATS.md`. MinHash is used for near-duplicate removal inside the training pool, not for the split check. We did not run an embedding-based check.

The check caught a real problem. When the upstream dataset changed and old training rows moved into `val` and `test`, our first SFT v2 build had 88 of 120 `val` rows and 84 of 120 `test` rows sharing field-name sets with training. A verification script found it because ids were missing. We rebuilt from a pinned revision. One caveat remains: about 78 to 81% of the answer strings in `val` and `test` also occur in the training answers, so these two splits flatter the model.

### CuratorKIT filtering
2,994 rows were pooled. CuratorKIT removed 140 near-duplicates (exact and MinHash), and its secrets gate rejected 22 rows (false positives on synthetic ids; we kept the gate on). That left 2,832 rows. A length cap and our split guard left 2,760. One more row was over 2,560 tokens, so training used 2,759.

### General capability benchmark
None was run. We have no check that fine-tuning did not reduce general ability (see Limitations).

## Metrics, and why these

| Metric | Direction | What it measures |
|---|---|---|
| `doc_exact` | higher is better | every field in the document is correct (the strictest metric) |
| `field_f1` | higher | a score per field value against the gold answer; correct nulls count |
| `null_acc` | higher | missing values are answered with `null` |
| `hallucination_rate` (raw and adjusted) | **lower is better** | values that the document does not support |
| `parse_ok`, `schema_valid` | higher | the output is valid JSON, and follows the schema |

`parse_ok` and `schema_valid` are **not** headline metrics: constrained decoding makes them near 100% for any model, so they say little about correct values. Our headline metrics are `field_f1`, `doc_exact`, and the gap between the in-domain splits and `test_b` and `test_c`. We test differences on the same documents (paired): a bootstrap gives a 95% interval, and an exact McNemar test covers `doc_exact`. All metric definitions live in one file (`eval.py`) and every script uses it. We do not use an LLM judge.

## Method and experiments

### CuratorKIT setup
Gates: schema validity, a length cap (about 2,300 estimated tokens), exact and MinHash deduplication, and the secrets gate. See "CuratorKIT filtering" for counts.

### AlignTune setup: SFT
QLoRA through `create_sft_trainer` (task type `sft`, TRL backend): 4-bit NF4 base, LoRA rank 32, alpha 64, dropout 0.05, all linear layers (60,456,960 trainable parameters), learning rate 1e-4 (cosine, 5% warmup), 2 epochs, batch 1 with gradient accumulation 16, maximum length 2,560, seed 42, fp16 on one free Kaggle T4. It took 346 optimizer steps in 4 hours 39 minutes. The loss covers only the JSON answer (about 14% of the tokens). The last training loss was about 0.003, which says nothing about new documents.

Problems that we found and fixed (each would have silently damaged the result):
- A T4 reports that it supports bf16 but emulates it, about four times slower. We use bf16 only on GPUs with compute capability 8 or higher.
- The trainer converts trainable weights to bf16 in its constructor, which crashes fp16 training. We restore fp32 when the optimizer is created.
- With task type `instruction_following`, AlignTune turns chat rows into plain strings and the chat template is never applied. We use `sft`.
- The Cohere chat template has no markers for the answer, so the loss covered every token. We patched it with `{% generation %}` markers. The rendered text is identical.

### AlignTune setup: DPO
`create_rl_trainer` (algorithm `dpo`, TRL backend, sigmoid loss), beta 0.1, LoRA rank 16, alpha 32, dropout 0.05, learning rate 1e-5 (untuned), 1 epoch, batch 1 with gradient accumulation 16 (89 steps), maximum length 2,048, bf16, seed 42. The SFT v2 adapter is merged into the base weights first, so that "adapter off" (AlignTune's reference) equals SFT v2. Check: the first-step loss was 0.6931 (= ln 2) with reward margin 0, so the policy started equal to the reference.

### AuditKIT setup
Greedy decoding, 1,280 new tokens, batch size 8, 4-bit model. A 384-token limit cut long answers; limits of 1,280, 1,536 and 2,048 gave identical outputs for the base model and the GRPO adapter. Every model name includes the token limit and the precision, because AuditKIT's run cache is keyed by the model name: early on, two different adapters returned identical scores from the cache.

### Seeds
Seed 42 everywhere. One run per configuration. We have no spread across seeds.

### Ablations
We did not run a controlled ablation. The comparison of the early adapter and SFT v2 changes the data, its size and its generators together, so it does not isolate any single choice.

## Results

### Main result: the SFT v2 adapter against the base model
Kaggle T4, fp16, all rows from one run, `verify_results.py` VERIFIED.

| split | system | parse_ok | field_f1 | doc_exact | null_acc | hallucination (adjusted) |
|---|---|---|---|---|---|---|
| `val` (120) | base | 1.000 | 0.971 | 0.608 | 0.955 | 0.004 |
| `val` | SFT v2 | 1.000 | 0.998 | 0.975 | 1.000 | 0.000 |
| `test` (120) | base | 1.000 | 0.962 | 0.533 | 0.939 | 0.004 |
| `test` | SFT v2 | 1.000 | 0.997 | 0.958 | 1.000 | 0.000 |
| `test_b` (415, synthetic) | base | 1.000 | 0.795 | 0.222 | 0.838 | 0.049 |
| `test_b` | SFT v2 | 0.983 | 0.815 | 0.263 | 0.873 | 0.045 |
| `test_c` (143, real) | base | 0.993 | 0.447 | 0.000 | 0.920 | 0.041 |
| `test_c` | SFT v2 | 0.993 | 0.494 | 0.000 | 0.962 | 0.027 |

### Confidence intervals and significance
Paired differences (SFT v2 minus base), 95% bootstrap intervals:
- `val` and `test`: doc_exact +0.367 and +0.425, significant.
- `test_b`: field_f1 +0.020 [+0.002, +0.038] and null_acc +0.036 are significant. **doc_exact +0.041 [-0.002, +0.084] is not significant** (McNemar p = 0.075). parse_ok fell from 1.000 to 0.983, a significant drop caused by broken JSON on some documents.
- `test_c`: field_f1 +0.047 [+0.022, +0.073] is significant. **No system gets a single document fully correct.**

In plain words: SFT v2 is far better on documents like its training data and somewhat better on a new domain and on real documents. It is not a solved problem on real documents.

### Finding 1: the earlier adapter memorized the templates
A teammate's run on another GPU (his own copy of the adapter, so compare only inside this table):

| split | base doc_exact | early SFT doc_exact | SFT v2 doc_exact |
|---|---|---|---|
| `val` | 0.550 | 0.983 | 0.975 |
| `test` | 0.517 | 0.983 | 0.975 |
| `test_b` | 0.210 | 0.010 | 0.246 |

The early SFT adapter (first dataset) scores 0.983 on documents that look like its training data and collapses to 0.010 on a new domain (parse_ok 0.839, field_f1 0.520). More varied data removes the collapse. Date normalization is a clear example: on `test_b` the base model leaves dates unnormalized 83.6 times per 100 documents (error count from our taxonomy tool), and SFT v2 0.0 times.

### Finding 2: what SFT v2 still gets wrong
Per 100 `test_b` documents: values turned into `null` 57.8 (base: 1.2), invented values 47.0, extra keys 96.6, wrong values 124.1, and 11% of documents break the schema. On `test_c` the null error is 107.7 per 100 documents (base 0.7), and extra keys are dominated by a few receipts.

### Sample input and output
A real `test_c` receipt, from the teammate's SFT v2 run (an excerpt of the document, then the gold answer and the model's answer):

```
RESTAURANT SIN DU / K3-113,JL IBRAHIM SULTAN / 80300 JOHOR BAHRU ... 09/03/2018 21:28 ...
DPT.05 RM 149.00 / DPT.04 RM 21.00 / CASH RM 170.00
Gold   : {"merchant_name": "RESTAURANT SIN DU", "date": "2018-03-09", "total_amount": 170.0, "subtotal": 170.0, "tax": 0.0, "currency": "MYR"}
Model  : {"merchant_name": "RESTAURANT SIN DU", "date": "2018-03-09", "total_amount": null, "subtotal": null, "tax": null, "currency": "RM"}
```
The model gets the name and the normalized date right, but writes `null` for the amounts and leaves the currency as `RM`. This is the over-null fault in one example. The gold values `subtotal 170.0`, `tax 0.0` and `MYR` are inferred or normalized, which is exactly where the model is weak.

## What did not work (negative results)

### GRPO
An adapter from an earlier GRPO attempt matched the base model within about one point on every metric and split (for example `val` doc_exact 0.340 for both, `test_b` doc_exact 0.227 against 0.224). That run used the first dataset, so its numbers are not comparable with the tables above. We did not find out why GRPO changed nothing.

### DPO on top of SFT v2
**Data.** 1,415 preference pairs (chosen = the gold JSON, rejected = a worse answer), 1,410 used. 247 rejected answers are real mistakes of SFT v2, sampled on 2,081 new prompts guarded against all training and evaluation sets. Only 277 of those 2,081 answers were wrong, because the prompts are easy. The other 1,168 come from rules (a value turned into `null`, an extra key, a wrong type, truncated JSON, and others), with the mix set from the error counts above. Both directions of the null error are included, so the model cannot learn "write more nulls". Training looked healthy: the loss fell from about 0.68 to 0.48 with no NaN (training numbers only).

**Result: the success rule fails.** One teammate GPU, greedy, 1,280 tokens, one precision for all rows, all files recomputed here and VERIFIED.

| `test_b` | parse_ok | field_f1 | doc_exact |
|---|---|---|---|
| SFT v2 adapter on the 4-bit base | 0.976 | 0.804 | 0.258 |
| merged SFT v2, no DPO (control) | 0.954 | 0.777 | 0.222 |
| merged SFT v2 plus DPO | 0.954 | 0.774 | 0.227 |

- DPO against the SFT v2 adapter looks worse (doc_exact -0.031 [-0.055, -0.007], field_f1 -0.029, significant).
- But DPO against the merged model that it started from is a tie: doc_exact +0.005 [-0.010, +0.019] (p = 0.75), field_f1 -0.002 [-0.015, +0.011]. On `test_c`, field_f1 is 0.491 for both. DPO changed 23% of the `test_b` answers with no net gain or loss.
- `val` and `test`: no significant change.

The control row is what makes this readable. Without it, the drop against the adapter would look like a DPO failure.

**Caveats.** The SFT v2 adapter rows and the merged rows were replayed from AuditKIT's cache of an earlier run on the same machine, not recomputed. Their outputs agree with the independent Kaggle run on 120 of 120 `val` and `test` answers and on 90% of `test_b` answers. The DPO rows were computed fresh. This is one run, one seed, an untuned learning rate, and few, easy on-policy pairs. It shows that this recipe did not help, not that DPO cannot.

### Merging the adapter into the weights
The merged model, loaded in 4 bit, is not bit-identical to the adapter on the 4-bit base. On `test_b` the merge costs 0.036 doc_exact [-0.058, -0.017] (McNemar p = 0.0007) and 0.027 field_f1. If you serve this model, keep it as an adapter on the 4-bit base.

**Decision:** the submitted model is the SFT v2 adapter. The DPO adapter is published as a documented negative result.

## Cost and ROI

### Compute used
- SFT v2 training: 4 hours 39 minutes on one free Kaggle T4.
- Evaluation: one system on the four splits takes hours on a T4 (`test_b` alone took 10,400 s for the adapter).
- DPO training and the DPO evaluation ran on a teammate's GPU. The GPU model, the training time and the hours were not reported to us.
- We know of no paid compute (free Kaggle tier and a teammate's GPU). We did not compute a money cost.

### Cost per 1,000 replies against an API
**Not done.** We measured speed only. On the Kaggle T4 the adapter handles about 0.04 documents per second (about 22,000 to 25,000 seconds per 1,000 documents), against 0.13 to 0.30 for the base model on the same setup. We did not investigate why the adapter is slower. On the teammate's GPU the DPO model handled about 0.08 documents per second. We have no closed-model output and no verified price source, so we quote no cost per 1,000 replies.

## Limitations

### Known weaknesses
- **Real documents are not solved.** `test_c` has no fully correct document for any system, and field_f1 is below 0.5. Typical faults: values turned into `null`, and extra keys (a few receipts produce very many).
- **New domain.** The doc_exact gain on `test_b` is not significant, parse_ok fell from 1.000 to 0.983, and `test_b` is synthetic, so it is a best case.
- **One run, one seed, one learning rate.** No variance estimate across seeds.
- **Not done:** base and SFT with constrained decoding, a closed-model reference, a cost comparison, a general-capability retention benchmark, and any non-English evaluation.
- **Synthetic, template-like, English training documents.** OCR noise, long text, tables and other languages are untested.
- **Hallucination rate.** The raw rate on `test_b` is higher than for the base model (0.253 against 0.201), while the adjusted rate is not different. We did not investigate this.
- **Speed.** The adapter is slow on a T4.
- **Licence.** CC-BY-NC-4.0 (non-commercial) like the base model. The paraloq note above is unresolved. Some upstream rows were written with Qwen2.5-7B-Instruct.
- **Data is not published.** The curated dataset and raw generation logs are not on Hugging Face (see Artifacts).

## With more credits

### Planned approach
1. **Real-document data:** add licensed, OCR-like receipts and forms to training, and a second real-document test set with a clear licence.
2. **A better DPO test:** on-policy pairs from hard prompts that resemble the new-domain faults, more pairs, a learning-rate and epoch sweep with several seeds, and serving the adapter on the 4-bit base so that the merge cost does not hide the result.
3. **Missing evaluations:** constrained decoding for the base model and the adapter, a closed-model reference with a real cost per 1,000 replies (tokens per second with vLLM), a general-capability retention benchmark, and multilingual tests.
4. **A larger student:** Aya Expanse 8B, which the rules allow.

### Expected before / after
We have no basis to forecast numbers, so we state targets, not forecasts. On `test_b`: a document exact match interval above the base model's. On `test_c`: at least some fully correct documents. On the in-domain splits: no loss.

## Artifacts

- **HF datasets collection:** **not published yet.** The curated dataset (`train_v2.jsonl`, 2,760 rows) and the raw generation logs stay local until the licence of the 151 paraloq rows is resolved. Planned: publish the dataset without those rows, with a card that says the trained model also saw them.
- **HF models collection:** https://huggingface.co/collections/monson2002/lexsi-hackathon
  - SFT v2 adapter (the submitted model): https://huggingface.co/monson2002/tiny-aya-sft-v2-adapter
  - DPO adapter (negative result): https://huggingface.co/monson2002/tiny-aya-sft-v2-dpo-adapter
- **GitHub repo:** https://github.com/Monson2002/lexsi-hackathon
- **Demo:** https://ghostinthegradient.vercel.app/ (the page states a T4 GPU that sleeps after 15 minutes, so the first request can be slow)
- **Article for the Hugging Face blog:** `docs/HF_ARTICLE.md`
- **Base model:** https://huggingface.co/CohereLabs/tiny-aya-global

## Documentation and reproduction

### Environment requirements
Python 3.11 or newer, a CUDA GPU with 16 GB or more for training and evaluation, and an HF token whose account accepted the terms of the gated base model. Pins: AlignTune 0.1.33 (TRL 1.7.1, PEFT 0.21.2, transformers 5.20.2), AuditKIT 1.1.6, CuratorKIT 0.1.24.

### Repo layout and run instructions
| Path | What it is |
|---|---|
| `eval.py` | AuditKIT evaluation harness and the single definition of every metric |
| `verify_results.py` | recomputes every metric from raw outputs; prints VERIFIED or NOT VERIFIED |
| `compare_results.py`, `error_taxonomy.py` | paired comparison with the DPO success rule; failure-type counts |
| `train_sft.py`, `smoke_test_sft.sh` | QLoRA SFT through AlignTune, and a short smoke test |
| `train_dpo.py`, `merge_sft_adapter.py`, `smoke_test_dpo.sh` | the DPO stage (merge SFT v2 into the weights, train, smoke test) |
| `fieldbench_to_jsonl.py` | converts FieldBench into the `test_b` and `test_c` evaluation files |
| `data/` | data builders: augmentation, schema-first generator, CuratorKIT build, leakage check, DPO pair builder, on-policy sampler. Generated data is not tracked (see the licence note) |
| `results/` | verified result tables, logs and paired comparisons |
| `sft_v2_adapter/` | model card, config and training log (the weights are on Hugging Face) |
| `docs/` | abstract, DPO plan and log, CuratorKIT usage notes, beginner walkthrough, article version |
| `AGENTS.md` | rules for contributors and coding agents |

The commands:

| Step | Command or file |
|---|---|
| Build the training data | `data/build_sft_v2.py` (reads the pinned dataset, runs CuratorKIT, applies the split guard, writes `STATS.md`) |
| Train SFT v2 | `python train_sft.py` (`--dry-run` first; `smoke_test_sft.sh` runs a short test) |
| Evaluate | `python eval.py --systems base sft_v2=<adapter> --splits val test test_b test_c --data-dir <folder>` |
| Verify | `python verify_results.py <results folder> --data-dir <folder>` |
| Compare | `python compare_results.py <results folder> base sft_v2` |
| DPO | `data/build_dpo_pairs.py`, `merge_sft_adapter.py`, `train_dpo.py`, `smoke_test_dpo.sh` (the first-step loss must be about 0.693) |

The plan and progress log of the DPO stage are in `docs/DPO_PLAN.md`, and a beginner walkthrough is in `docs/PROJECT_EXPLAINER.html`.

### Declaration of work done outside the hackathon
No part of the submitted work was done before or outside the hackathon. External material that we used: the base model, the public datasets listed above (JSONSchemaBench, paraloq, FieldBench) and the hackathon toolkits.

## Discussion: what we learned
- An in-domain score of 0.98 can hide total failure on a new domain: judge by the new-domain and real-document numbers.
- A control row can reverse a conclusion. Without the merged-model control, "DPO hurts" would have been published, while the truth is "DPO is neutral and our merge step hurts".
- Recomputing every metric from raw outputs, and naming runs with their token limit and precision, caught cache and leakage bugs early.

## Teammates tagged
- Monson Reji Verghese, Hugging Face `monson2002`
- Sujal Gawas, Hugging Face `sujalgawas`
- James, Hugging Face handle: *to be added*

## Author contributions (to be confirmed by the team)
- Sujal: the original dataset and notebook, and the early SFT and GRPO adapters.
- James: GPU runs on his machine (SFT v2 evaluation, the on-policy sampling, DPO training and evaluation).
- Monson: the evaluation harness and verification tools, the SFT v2 data rebuild and training, the DPO pipeline, the Hugging Face repos and the write-up.

## Acknowledgements
Free Kaggle GPU quota, and a teammate's college GPU.

## References
Tiny Aya (arXiv 2603.11510), JSONSchemaBench (arXiv 2501.10868), SO-Bench (arXiv 2511.21750), AlignTune (arXiv 2602.09621), CuratorKIT (arXiv 2606.21631). FieldBench corpus: https://github.com/fieldbench/corpus.

## LLM usage statement
All development was done with LLM assistance: the code (data builders, training and evaluation scripts, verification tools), the data-generation logic and this write-up were produced with Claude Code (Anthropic) under human direction. Humans ran the GPU jobs, supplied the data and results, and made the decisions. Some rows of the upstream dataset were written with Qwen2.5-7B-Instruct, as noted above.

## Changelog
- Nov 16, 2026: this write-up is now the README (the earlier README, with the dated results by run, is kept as the appendix below).
- Nov 16, 2026: restructured to the recommended write-up layout. Corrected the description of `test_b` (415 synthetic documents) and `test_c` (143 real documents, 131 of them receipts). Added the limitations on language and capability retention.

---

## Appendix: all results by run, with dates (status on Nov 15, 2026)

Every number below comes from a result file in this repo (`results/`) or in the model repo [`monson2002/tiny-aya-sft-v2-adapter`](https://huggingface.co/monson2002/tiny-aya-sft-v2-adapter) under `results/`. Nothing is estimated. DPO was trained and evaluated: it gave **no gain** over SFT v2 (see the DPO section).

### How to read the numbers (for beginners)
- **Splits.** `val` and `test` (also called `test_a` in the oldest run) hold documents like the training ones, so scores there are high and flatter the model. `test_b` is a whole new domain (415 synthetic documents from FieldBench). `test_c` is 143 real documents, mostly receipts. The honest test of "did it learn the general skill" is `test_b` and `test_c`.
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
- `test_c`: field_f1 +0.047 [+0.022, +0.073] is significant. No system gets a single document fully correct.

**In plain words:** SFT v2 is much better on documents like its training data and somewhat better on new domains and real documents (mostly receipts). It is not a solved problem on real documents.

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
