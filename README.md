# lexsi-hackathon

Fine-tuning a small Aya model (Tiny Aya Global, 3.35B) for **schema-conditioned structured extraction**: the input is a document plus a JSON Schema, the output is one JSON object that follows the schema, with `null` for values that are not in the document.
Lexsi Labs hackathon, Track 1 (fine-tuning and data curation). Toolkits chained: CuratorKIT (data), AlignTune (training), AuditKIT (evaluation). Deadline: Nov 14, 2026, end of day (AoE).
A beginner-friendly walkthrough of the theory and the full history is in [`docs/PROJECT_EXPLAINER.html`](docs/PROJECT_EXPLAINER.html). The DPO plan is in [`docs/DPO_PLAN.md`](docs/DPO_PLAN.md).

## Results so far (status on Nov 13, 2026)

Every number below comes from a result file in this repo (`results/`) or in the model repo [`monson2002/tiny-aya-sft-v2-adapter`](https://huggingface.co/monson2002/tiny-aya-sft-v2-adapter) under `results/`. Nothing is estimated. Nothing exists yet for DPO.

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

### What we know and do not know
- **Known:** SFT v2 fixed the main failure of the early SFT (collapse on a new domain). Date normalisation, markdown fences and missing-field handling improved a lot. The GRPO adapter did nothing measurable.
- **Not known:** whether DPO helps (not run yet). Why the SFT v2 evaluation was slow (0.04 documents per second on the T4, and 12.9 seconds per document on the teammate's GPU).
- **Remaining weaknesses seen in the error counts of run B:** the model writes `null` where the document has a value (64 of 100 documents on `test_b`, against 1.4 for base), makes bracket errors in nested JSON, and puts a list into a string field.
- **Open licence question:** 151 of the 2,760 training rows come from `paraloq/json_data_extraction`, whose licence terms are unclear. The team has not decided yet.

### Rerun the checks
`python verify_results.py <results_dir>` recomputes every metric from the raw outputs and prints VERIFIED or NOT VERIFIED. `python compare_results.py <results_dir> base sft_v2_hf --no-rule` gives the paired comparison.
