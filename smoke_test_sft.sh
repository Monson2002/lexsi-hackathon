#!/usr/bin/env bash
# Smoke test for SFT v2, about 10-15 min on one GPU. Run from the repo root (train_sft.py, eval.py, data/sft_v2/ next to each other).
#   HF_TOKEN=hf_xxx bash smoke_test_sft.sh
# Proves: AlignTune loads our chat file, the loss mask is right, 5 real QLoRA steps run and save, AuditKIT can load and score the adapter.
# It does NOT prove the model got better: 5 steps on 64 rows trains almost nothing. Note the seconds per step: full run = ~346 steps.
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN (CohereLabs/tiny-aya-global is gated)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # one GPU; the 4-bit model fits a T4 and splitting it only slows it down
OUT=smoke_sft_v2

echo "== 1/3 dry run (no GPU needed): template check, loss mask, training config"
python train_sft.py --dry-run --max-samples 64 --out "$OUT/dry"

echo "== 2/3 five real training steps on 64 rows"
python train_sft.py --max-samples 64 --max-steps 5 --out "$OUT/train"

echo "== 3/3 score the smoke adapter on 10 val documents (checks the adapter loads in eval.py and the output parses)"
python eval.py --dataset sujalgawas/aya-json-sft-dataset --splits val --limit 10 \
  --systems "sft_smoke=$OUT/train/adapter" --out "$OUT/eval"
cat "$OUT/eval/results.md"
echo "smoke test passed: check seconds/step in the training log above, then run: python train_sft.py"
