# lexsi-hackathon

Fine-tuning a small Aya model (Tiny Aya Global, 3.35B) for **schema-conditioned structured extraction**: the input is a document plus a JSON Schema, the output is one JSON object that follows the schema, with `null` for values that are not in the document.
Lexsi Labs hackathon, Track 1 (fine-tuning and data curation). Toolkits chained: CuratorKIT (data), AlignTune (training), AuditKIT (evaluation). Deadline: Nov 14, 2026, end of day (AoE).
A beginner-friendly walkthrough of the theory and the full history is in [`docs/PROJECT_EXPLAINER.html`](docs/PROJECT_EXPLAINER.html). The DPO plan is in [`docs/DPO_PLAN.md`](docs/DPO_PLAN.md). The write-up is [`docs/WRITEUP.md`](docs/WRITEUP.md) and the abstract is [`docs/ABSTRACT.md`](docs/ABSTRACT.md).

## Results so far (status on Nov 15, 2026)

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
