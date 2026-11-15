#!/usr/bin/env bash
# Smoke test for the DPO stage, about 15-20 min on one GPU. Run from this folder.
#   HF_TOKEN=hf_xxx SFT_ADAPTER=monson2002/tiny-aya-sft-v2-adapter bash smoke_test_dpo.sh   (or a local adapter folder)
# Proves: SFT v2 merges into the base, the pairs render to the right strings, 3 real DPO steps run, the FIRST-STEP LOSS IS ln 2 (= 0.693: policy and
# reference start equal), the adapter loads in eval.py on top of the merged model and its output parses. It does NOT show that DPO helped.
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN (CohereLabs/tiny-aya-global is gated)}"
: "${SFT_ADAPTER:?set SFT_ADAPTER to the SFT v2 adapter folder or HF repo}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PAIRS="${PAIRS:-dpo_pairs_train.jsonl}"
OUT=smoke_dpo

echo "== 1/4 merge the SFT v2 adapter into the base weights (skipped if merged_sft_v2 exists)"
[ -f merged_sft_v2/config.json ] || python merge_sft_adapter.py --adapter "$SFT_ADAPTER" --out merged_sft_v2

echo "== 2/4 dry run: pair rendering, token-level check against the chat template, training config"
python train_dpo.py --model merged_sft_v2 --pairs "$PAIRS" --dry-run --max-samples 64 --out "$OUT/dry"

echo "== 3/4 three real DPO steps on 32 pairs"
python train_dpo.py --model merged_sft_v2 --pairs "$PAIRS" --max-samples 32 --max-steps 3 --grad-accum 4 --out "$OUT/train" 2>&1 | tee "$OUT/train.log"
if grep -q "WARNING: the first-step loss" "$OUT/train.log"; then echo "SMOKE TEST FAILED: policy and reference did not start equal"; exit 1; fi
grep -E "first logged step|trainable dtypes at optimizer creation" "$OUT/train.log" || true

echo "== 4/4 score the smoke adapter on 10 val documents, on top of the merged model (checks it loads in eval.py and the output parses)"
python eval.py --data-dir . --splits val --limit 10 --systems "dpo_smoke=$OUT/train/adapter@merged_sft_v2" --out "$OUT/eval"
cat "$OUT/eval/results.md"
echo "smoke test passed: note seconds per step in the log above"
