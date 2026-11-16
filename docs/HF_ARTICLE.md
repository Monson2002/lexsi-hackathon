# What we learned teaching a 3B model to follow any JSON Schema (and why DPO did nothing)

*Lexsi Labs hackathon, Track 1: fine-tuning and data curation. We fine-tuned Tiny Aya Global with CuratorKIT, AlignTune and AuditKIT. This post is the honest version: what worked, what did not, and what we did not measure.*

## TL;DR

- **Task:** give the model a document and a JSON Schema. It must return one JSON object that follows the schema, with values taken from the document and `null` for anything missing.
- **What worked:** a QLoRA adapter trained on 2,760 carefully built examples. It raises document exact match from 0.608 to 0.975 on in-domain validation and from 0.533 to 0.958 on in-domain test documents.
- **What is still hard:** a new domain and real documents (mostly receipts). On a new domain, document exact match goes from 0.222 to 0.263, and that gain is **not** statistically significant. On 143 real documents, mostly receipts, no system, ours or the base model, gets a single document fully correct.
- **What did not work:** a GRPO adapter (no change), DPO on top of our adapter (no measurable effect), and merging the adapter into the base weights (a significant loss, about 3.6 points of document exact match on the new domain).
- **Everything is open:** adapters, code, verified result tables and a demo. Links are at the end.

## 1. The task, and why "any schema" is the hard part

Structured extraction sounds easy until you ask for the general skill. A model that fills in one fixed form is not useful. We want a model that follows whatever schema it is given: nested objects, arrays, enums, optional fields, and fields that look related but are absent from the document.

So the model must follow four rules:

1. The output is valid JSON and follows the schema.
2. Every value comes from the document. The model never invents a value.
3. A missing value is `null`.
4. The model changes a value (for example a date) only when the schema asks for it.

We test three things the model never saw in training: new documents with known schemas (`val`), new schemas (`test`), and a whole new domain (FieldBench `test_b`: 415 synthetic documents from insurance, contracts, legal and SEC filings). A small set of real documents (mostly receipts) (`test_c`) checks real-world text.

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
| `test_b`, new domain (415 synthetic) | base | 1.000 | 0.795 | 0.222 | 0.838 |
| `test_b` | **adapter** | 0.983 | 0.815 | 0.263 | 0.873 |
| `test_c`, 143 real documents, mostly receipts | base | 0.993 | 0.447 | 0.000 | 0.920 |
| `test_c` | **adapter** | 0.993 | 0.494 | 0.000 | 0.962 |

What the paired tests say:

- In domain, the gains are large and significant (+0.367 and +0.425 document exact match).
- On the new domain, field F1 (+0.020) and null accuracy (+0.036) improve significantly. **Document exact match (+0.041, interval -0.002 to +0.084) does not.** The parse rate also fell from 1.000 to 0.983, because of broken JSON on a few documents.
- On real documents (mostly receipts), field F1 improves (+0.047, significant), but **document exact match is 0.000 for every system**.

The adapter fixed the failures that were easy to fix: date normalization, markdown fences around the JSON, and missing-field handling. It did not make the model good at messy, real documents. We think that is the honest summary.

## 7. Three things that did not work

### GRPO
An earlier GRPO adapter matched the base model within about one point on every metric and split. We did not find out why.

### DPO, and the control that saved us
SFT left new faults. The model now turns real values into `null` (58 and 108 times per 100 documents on the new domain and on `test_c`, against 1.2 and 0.7 for the base model), adds extra keys, invents values, and sometimes writes broken JSON.

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

- **`test_b` is synthetic.** FieldBench's own datasheet warns that synthetic documents overestimate accuracy, so read the `test_b` numbers as a best case for a new domain.
- **Real documents are not solved.** Document exact match on `test_c` is 0.000, and field F1 is below 0.5.
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
