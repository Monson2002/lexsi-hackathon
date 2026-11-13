# Plan: add a DPO stage on top of SFT v2 and compare

Written 2026-11-12. Deadline: Nov 14, 2026 end of day (AoE). Status of every claim: **[verified]** = checked in code or a run, **[estimate]** = a guess to be replaced by a measurement,
**[unverified]** = from memory or one source, must be tested before relying on it.

## 1. Goal, hypothesis, success rule

**Goal.** Test whether a short preference-optimisation stage (DPO) on top of the SFT v2 adapter fixes the failures SFT alone leaves, and report the before/after honestly.

**Hypothesis.** SFT teaches the format but not the rules that matter on new domains (dates must be normalised, required keys must be present, absent values must be `null`, no extra keys, no schema echo, enum labels
only from the list). DPO shows the model a correct and an incorrect answer for the same prompt, which is a more direct signal for exactly those rules. **[unverified]**: one preprint (RL-Struct, a 4B model, a recipe-extraction task)
reports DPO above SFT on JSON validity; nothing says it transfers.

**Systems compared, one session, same caps (1280 new tokens), same four splits (val 120, test 120, test_b 415, test_c 143):**
`base`, `sft` (old adapter), `sft_v2`, `sft_v2_merged` (control, see section 3b), `sft_v2_dpo`. The comparison that decides this plan is `sft_v2` vs `sft_v2_dpo`.

**Success rule, fixed before any result exists** (so we cannot move the goalposts):
- "DPO helps" = on `test_b` AND `test_c`, `doc_exact` and `field_f1` of `sft_v2_dpo` are above `sft_v2`, the paired difference is clearly outside noise (section 6), AND on `val`/`test` no metric drops by more than the noise band, AND `parse_ok` and `schema_valid` do not fall.
- Anything else is reported as "no reliable gain" and the submission uses `sft_v2` alone. AGENTS.md: an SFT-only submission done well is valid; RL/preference work must not block the deadline.

## 2. Facts the plan stands on

| Fact | Status |
|---|---|
| AlignTune 0.1.33 has `TRLDPOTrainer` (`create_rl_trainer(algorithm="dpo", backend="trl")`), config defaults beta 0.1, loss `sigmoid`; extra TRL options such as `precompute_ref_log_probs` pass through | **[verified]** (source) |
| With PEFT/4-bit on, AlignTune's DPO sets `ref_model=None`, so TRL uses the model with the adapter switched OFF as the reference. A fresh LoRA on the base model therefore has the BASE as reference, not SFT v2 | **[verified]** (source) |
| AlignTune has no direct hook to start from an existing adapter and use it as the reference | **[unverified]**: needs the prototype in WP4 |
| CuratorKIT ingests `chosen`/`rejected`, applies its gates, exports `dpo.jsonl`, but the exported `prompt` is only the instruction field: the document and schema are lost | **[verified]** (12-row probe) |
| TRL 1.7.1's `SFTTrainer` casts all trainable weights of a quantised model to bf16, which crashed fp16 on a T4; `DPOTrainer` may do the same | **[unverified]**: test in the GPU smoke test |
| SFT step time on a Kaggle T4 in fp16: 57 s per optimizer step (16 sequences of about 1.2K tokens) | **[verified]** |
| DPO processes a chosen AND a rejected sequence per pair | **[verified]** (by definition), so expect roughly 2x the time per pair |

## 3. Prerequisites and gates (nothing in WP5 starts before Gate 0 and Gate 1)

- **Gate 0:** the SFT v2 adapter exists, was evaluated at 1280 on all four splits, and `verify_results.py` says VERIFIED.
- **Gate 1:** the failure analysis of SFT v2 (WP0) shows failures that preference pairs can address (section 4, WP0). If SFT v2 already scores well on `test_b`/`test_c`, or its remaining errors are wrong extractions rather than format/rule errors, stop here and do not run DPO.
- **Gate 2:** the DPO smoke test (5 steps) passes on a GPU: loss is a number, no dtype crash, memory fits.
- **Gate 3 (hard cut-off):** the DPO adapter must exist by the evening of Nov 13, with a full evaluation still to run. If not, drop DPO.

## 3b. Progress log, 2026-11-12 (CPU-only work done while SFT v2 trains)

Built and tested (all on CPU, with a tiny random Cohere2 model for anything that trains; the GPU parts are untested):

| Item | File | State |
|---|---|---|
| Paired comparison + the success rule of section 1 | `compare_results.py` | tested on base vs GRPO: correctly says "no reliable gain" |
| Failure-type counter (dates, missing keys, echo, nulls ...) | `error_taxonomy.py` | tested on base results; already shows base leaves 86 dates per 100 documents unnormalised on `test_b`, and 57% of its `test_c` answers wrapped in ``` fences |
| Fresh prompt pool from JSONSchemaBench `val`/`test` schemas, guarded against SFT training and every evaluation set | `data/gen_schema_first.py --split`, `data/build_dpo_pairs.py prompts` | 2,483 generated, 2,081 kept (402 dropped for sharing a field-name set with SFT training), 1,278 distinct schemas, 0 overlap with any evaluation set |
| Rule-based pairs with per-kind quotas, CuratorKIT pass, final export | `data/build_dpo_pairs.py build` | 1,163 pairs from 925 prompts; date pairs 216 (was 33 before quotas), enum 96; every rejected answer checked to score below the gold |
| On-policy sampler (resumable) and its merge into the pairs | `data/sample_onpolicy.py`, `build ... --onpolicy` | stub-tested, including resume; the GPU path shares the loading code of `eval.py` but has not run |
| Start from SFT v2 with reference = SFT v2 | `merge_sft_adapter.py`, `data/dpo_start_check.py` | verified: first-step DPO loss 0.6931 = ln 2, reward margin 0 |
| DPO training script | `train_dpo.py`, `smoke_test_dpo.sh` | dry run and 2 real steps on CPU pass; GPU untested |
| Evaluation of an adapter on a merged base | `eval.py`: `name=<adapter>@<base folder>` and `name=<merged folder>` | stub and parsing tested |

Findings that changed the design (each verified in code or on the real tokenizer):
1. **AlignTune's data loader flattens message lists to plain strings** before TRL sees them (system prompt glued to the user text, no chat template). Passing our chat-format pairs as they are would train DPO on a different prompt format than we evaluate with. `train_dpo.py` renders the template into plain strings itself: the leading BOS is stripped (TRL adds one), the chosen answer ends with `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>` which is already the tokenizer's EOS, and a check proves that TRL's tokenization equals the chat-template ids: **600 of 600** sequences on the real tokenizer, and the run aborts if it ever does not.
2. **TRL's DPO defaults would cut our examples:** `max_length=1024`, `truncation_mode=keep_start` (the answers are at the end). Set explicitly to 2048, and pairs that do not fit are dropped and counted.
3. **TRL's DPO defaults to bf16** (a CPU refuses it, a T4 would emulate it slowly), and `DPOTrainer` casts the trainable weights of a quantised model to bf16 just like `SFTTrainer`. Fixed with module-level subclasses (same technique as SFT): bf16 only on Ampere or newer, fp32 trainable weights otherwise.
4. **Merge-then-new-LoRA is the chosen starting point (WP4 option 1)**, verified by the ln 2 test. It adds a confound: the merged weights are quantised again, so "merged SFT v2" is not bit-identical to "4-bit base + SFT adapter". The comparison therefore gets a **control row `sft_v2_merged`** (the merged model alone, no DPO): if it differs from `sft_v2`, that is the merge effect, not DPO. The success rule stays as accepted (`sft_v2` vs `sft_v2_dpo`); `sft_v2_merged` is reported as the attribution check.

**Phases (James's GPU is needed for 1 and 3):**
1. *Sample:* James runs `sample_onpolicy.py` on `pool_clean.jsonl` with the SFT v2 adapter **`monson2002/tiny-aya-sft-v2-adapter`, pinned to commit `9774195260373ce418e3e3869cd3c7bef84f47b8`** (weights SHA-256 `39cfb015...`, decided 2026-11-13), and sends back `onpolicy_samples.jsonl` plus its `.meta.json` (small). The merge, the DPO run and the evaluation must use that same adapter and commit.
2. *Pairs:* I build the final pairs from the samples plus the rule-based pairs, using the failure counts of `error_taxonomy.py` as weights, and send James `dpo_pairs_train.jsonl`.
3. *Train and evaluate:* James merges, runs `smoke_test_dpo.sh`, then trains and evaluates `base`, `sft`, `sft_v2`, `sft_v2_merged`, `sft_v2_dpo`.
Phases 1 and 2 need the SFT v2 adapter and its evaluation (Gate 0 and Gate 1).

## 4. Work packages

**WP0. SFT v2 failure analysis (me, when SFT v2 results arrive, about 1 hour).**
Count, from the raw outputs of `sft_v2` per split: date left unnormalised, required key missing, extra key, schema echo, enum outside the list, invented value where gold is `null`, `null` where gold has a value, type errors, unfinished JSON. The counts set the mix of pair kinds in WP2. New script: `error_taxonomy.py` (also useful for the write-up).

**WP1. Fresh prompt pool (me, about 2 hours, no GPU).**
The SFT model has seen its training prompts, so it makes few mistakes on them; pairs built there would be nearly all synthetic. Use prompts it has **not** seen:
- new documents from the schema-first generator with a new seed;
- schemas from the JSONSchemaBench `val`/`test` splits (the SFT run used only the `train` splits), **[unverified]** that they are disjoint from the ones used, check with the leakage script;
- the existing leakage guard against `val`, `test`, `test_b`, `test_c` is mandatory. Never build pairs from the evaluation sets.

**WP2. Rejected answers: two sources.**
- *(a) Rule-based corruptions of the gold* (me, no GPU, deterministic): one failure per pair, chosen = gold. Kinds follow WP0, for example: date written as in the document; required key dropped; extra key added; schema fragment echoed into the answer; invented value where gold is `null`; enum label outside the list; wrong type; fenced or truncated JSON. **Balance both directions**: include pairs where the rejected answer has a `null` where gold has a value (missed value), otherwise the model can learn "write more nulls" (the analogue of the all-null reward hack in AGENTS.md). Check afterwards that `null_acc` and recall do not trade off.
- *(b) On-policy failures* (needs a GPU, James or Kaggle): sample `sft_v2` on the fresh prompts (1 to 2 samples, temperature about 0.7), score against gold with the shared metric, keep samples that score below the gold (rejected) and discard correct ones. Cost: one generation pass over the pool; **[estimate]** a few seconds per document on a T4 in fp16, to be measured.
- Target mix **[estimate]**: about 60% on-policy, 40% rule-based; if the GPU time is not available, run with rule-based only and say so in the write-up. Target size **[estimate]**: 1,500 to 3,000 pairs.

**WP3. CuratorKIT pass and export (me, about 2 hours).**
Pool the pairs, run CuratorKIT (ingest, schema gate, exact and MinHash dedup, secrets gate, provenance files) exactly as for SFT, then write the final training file ourselves from the surviving ids in the **frozen prompt format as chat messages**: prompt = [system, user], chosen = [assistant], rejected = [assistant]. We do not use CuratorKIT's `dpo.jsonl` because it drops the document and schema. Update `docs/CURATORKIT_USAGE.md` with the counts. Apply the same split guard as the SFT build.

**WP4. Starting point: policy = SFT v2, reference = SFT v2 (me, about 2 hours, CPU, tiny Cohere2 model).**
Options to prototype, in order of preference:
1. *Merge then new LoRA:* merge the SFT v2 adapter into the base weights, save the merged model, then run AlignTune DPO with `model_name` = the merged model and a fresh LoRA. Reference (adapter off) = merged SFT v2. Concern: the merged weights are then quantised to 4 bit, which is not identical to "4-bit base plus fp adapter" as evaluated. Measure that difference by evaluating the merged model on a slice.
2. *Dual adapter in TRL* (`model_adapter_name` / `ref_adapter_name`): exact, but it bypasses AlignTune's trainer wrapper; only if option 1 fails, and it must still be launched through AlignTune.
**Sanity test that proves the setup (run on the tiny model first, then in the GPU smoke test):** at step 0 the DPO loss must be about 0.693 (ln 2) and reward margins about 0, because policy equals reference. Any other value means the reference is wrong.

**WP5. `train_dpo.py` through AlignTune (me).**
Same structure as `train_sft.py` (dry run, smoke test, one flag to push privately), plus the dtype fix if the GPU smoke test shows the bf16 crash. Starting hyperparameters **[unverified, to be tuned only if time allows]**: beta 0.1 (AlignTune default), loss `sigmoid`, 1 epoch, small learning rate for a LoRA, LoRA r 16, `max_length` 2048, gradient checkpointing, `precompute_ref_log_probs` on if it saves memory. Log per step: loss, reward margin, reward accuracy.

**WP6. GPU smoke test then full DPO run (James or Kaggle).** Gate 2 above. Time **[estimate]**: about 2x the SFT per-example time, so a few hours for a few thousand pairs.

**WP7. Evaluation and comparison (James runs, I verify).**
```
python eval.py --dataset sujalgawas/aya-json-sft-dataset --data-dir . --splits val test test_b test_c \
  --systems base sft sft_v2=<sft_v2 adapter> sft_v2_dpo=<dpo adapter> --out res_dpo
python verify_results.py res_dpo --adapter <dpo adapter>
python compare_results.py res_dpo sft_v2 sft_v2_dpo          # new: paired comparison, see section 6
```
Same session, cap 1280, pinned `val.jsonl`/`test.jsonl`. Report our six metrics plus AuditKIT's `ak_*` columns and speed.

**WP8. Decision and write-up (me and you).** Apply the success rule. Report the table including failures and the per-failure-type counts. If DPO does not help, say so and ship `sft_v2`.

## 5. Files this adds

`error_taxonomy.py` (WP0), `data/build_dpo_pairs.py` (WP1 to WP3), `train_dpo.py` (WP5), `smoke_test_dpo.sh`, `compare_results.py` (WP7), a second handoff bundle for James, and `docs/DPO_RESULTS.md` after the run. All with the same checks as SFT: syntax, CPU tiny-model run, clean-unzip test, checksums.

## 6. How we decide it is not noise

Per-document results exist for every system (AuditKIT stores them), so the comparison is **paired**: the same document under both systems. `compare_results.py` will give, per split: the difference in each metric with a paired bootstrap 95% interval, and for `doc_exact` an exact McNemar test on the discordant documents. Rough scale **[estimate]**: with 415 documents a single proportion has a standard error of about 2.4 points, so a gain below about 5 points on `test_b` is not distinguishable from noise unless the paired test says so; `test_c` has 143 documents and is noisier, and receipts make `doc_exact` near 0 there, so use `field_f1` and `null_acc` on `test_c`.
Also check for collapse: output length, `parse_ok`, `schema_valid`, share of all-null outputs.

## 7. Schedule against the deadline (today is Nov 12)

| When | What |
|---|---|
| now, while SFT v2 trains | WP1 and the rule-based part of WP2, WP3 scaffolding, WP4 prototype on the tiny model, `error_taxonomy.py`, `compare_results.py` (all CPU, no results needed) |
| SFT v2 adapter ready (Kaggle about 5.5 h from the commit start, or James) | evaluate it (Gate 0), run WP0 (Gate 1) |
| Nov 13 morning | on-policy sampling (WP2b), build and curate the pairs, GPU smoke test (Gate 2) |
| Nov 13 daytime | full DPO run |
| Nov 13 evening | **Gate 3.** No DPO adapter by now means DPO is dropped |
| Nov 13 night to Nov 14 | evaluation, verification, comparison, write-up |
| Nov 14 | submission: models and datasets public on the Hub, repo public, README, abstract, demo |

## 8. Risks and stop rules

- DPO can lower `parse_ok`/`schema_valid` or make outputs shorter or odder: discard if `val`/`test` regress beyond noise.
- "Write more nulls" drift: watch `null_acc` against `field_f1`; rebalance pairs once, not repeatedly (that would be tuning on the evaluation sets).
- Off-policy rule-based pairs can be too easy (margin saturates immediately): the reward accuracy in the log will show it; then rely more on on-policy pairs.
- Memory at 2K tokens with two sequences per pair on a 16 GB card: use batch 1 with accumulation; shorten with `--max-length` if needed.
- Time: if WP4 (starting point) is not solved by the morning of Nov 13, drop DPO. Never tune on `val`/`test`/FieldBench.
- Licence: the paraloq rows (151) stay out of the DPO pool unless the team resolves the licence note.

## 9. Decisions (Monson, 2026-11-12)

1. DPO training and the on-policy sampling run on **James's GPU**.
2. On-policy sampling (WP2b) is **in**; it costs about one extra GPU pass.
3. The success rule in section 1 is **accepted as written**.
4. System names stay `sft_v2` and `sft_v2_dpo` (assumed, nobody objected).
