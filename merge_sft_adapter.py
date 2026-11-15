"""Merge the SFT v2 LoRA adapter into the base weights and save a plain model (the starting point for DPO, docs/DPO_PLAN.md WP4).

    python merge_sft_adapter.py --adapter sft_v2_adapter/adapter --out merged_sft_v2          # loads the base in 16 bit, about 7 GB of RAM or GPU memory

Why: AlignTune's DPO trainer uses the model with the adapter switched OFF as the reference. A fresh LoRA on the base would therefore have the BASE as reference.
With the SFT adapter merged into the weights, "adapter off" = SFT v2, so the DPO reference is SFT v2 (checked: step-0 DPO loss = ln 2, data/dpo_start_check.py).
Also saves the base tokenizer with its chat template as ONE plain string (the base stores a dict of named templates; the default entry is what eval.py and the
trainers use), so everything that loads the merged folder renders prompts exactly as before.
Caveat to measure: the merged weights are quantised to 4 bit again when loaded, which is not bit-identical to "4-bit base + adapter" as evaluated. Evaluate the merged
model as its own row (eval.py --systems sft_v2_merged=<this folder>) next to sft_v2, so that effect is visible and not mistaken for a DPO effect.
"""
import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="SFT v2 adapter folder or HF repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="the base checkpoint is bf16; keep it")
    ap.add_argument("--device", default="cpu", help="cpu (slow, safe) or cuda")
    ap.add_argument("--template-from", help="tiny-model tests only: repo whose tokenizer carries the Cohere chat template")
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, token=token, device_map=args.device)
    peft_model = PeftModel.from_pretrained(model, args.adapter)
    probe = torch.tensor([[1, 5, 9, 14, 3, 8, 2, 7]], device=model.device)
    with torch.no_grad():
        before = peft_model(probe).logits.float()
    merged = peft_model.merge_and_unload()
    with torch.no_grad():
        after = merged(probe).logits.float()
    diff = float((before - after).abs().max())
    scale = float(before.abs().max())
    print(f"max |logit difference| adapter-on vs merged on a probe input: {diff:.4g} (logit scale {scale:.3g})")
    if diff > 0.05 * max(scale, 1.0):
        raise SystemExit("merge changed the output too much: do not use this merged model")
    out = Path(args.out)
    merged.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base, token=token)
    if args.template_from:
        tok.chat_template = AutoTokenizer.from_pretrained(args.template_from).chat_template
    if isinstance(tok.chat_template, dict):
        tok.chat_template = tok.chat_template["default"]
    tok.save_pretrained(out)
    print("saved merged model + tokenizer to", out)


if __name__ == "__main__":
    main()
