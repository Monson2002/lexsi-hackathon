"""Does DPO start from SFT v2 with reference = SFT v2?  (docs/DPO_PLAN.md WP4)  CPU, tiny random Cohere2 model, a few minutes.

    python data/dpo_start_check.py --adapter /tmp/at_smoke4/adapter --pairs data/dpo/dpo_pairs_train.jsonl

Why it matters: AlignTune's DPO trainer uses the model with its LoRA adapter switched OFF as the reference (`ref_model=None` when PEFT is on). If we trained a
fresh LoRA on the BASE model, the reference would be the base, not SFT v2. So: merge the SFT v2 adapter into the base weights, save that as a plain model,
and run DPO on it with a fresh LoRA. Policy (merged + zero-initialised LoRA) and reference (merged, adapter off) are then the same function, so at step 0
    DPO loss = ln 2 = 0.6931   and   reward margin = 0   and   reward accuracy = 0.5 (ties)
Any other first-step values mean the reference is not what we think. The script checks exactly that, then runs the AlignTune DPO trainer for two steps.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import trl


class CpuDPOConfig(trl.DPOConfig):
    def __post_init__(self):
        self.bf16, self.fp16, self.use_cpu = False, False, True
        super().__post_init__()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="hf-tiny-v2/tiny-random-Cohere2ForCausalLM")
    ap.add_argument("--adapter", required=True, help="an SFT adapter trained on that base (e.g. the one train_sft.py saved in a tiny-model smoke test)")
    ap.add_argument("--pairs", required=True, help="dpo_pairs_train.jsonl (prompt/chosen/rejected as message lists)")
    ap.add_argument("--template-from", default="sujalgawas/aya-json-sft", help="repo whose tokenizer carries the Cohere chat template (the tiny model's own has none)")
    ap.add_argument("--workdir", default="/tmp/dpo_start_check")
    ap.add_argument("--n", type=int, default=8, help="pairs to use")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work = Path(args.workdir)
    shutil.rmtree(work, ignore_errors=True)
    merged = work / "merged"
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(merged)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.chat_template = AutoTokenizer.from_pretrained(args.template_from).chat_template  # tiny model only; the real base already has it
    if isinstance(tok.chat_template, dict):
        tok.chat_template = tok.chat_template["default"]
    tok.save_pretrained(merged)
    print("merged SFT adapter into the base and saved it:", merged)

    pairs = work / "pairs_small.jsonl"
    with open(args.pairs, encoding="utf-8") as f, open(pairs, "w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            r = json.loads(line)
            g.write(json.dumps({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}, ensure_ascii=False) + "\n")

    # AlignTune builds TRL's DPOConfig from a fixed argument list that has no precision, and TRL then defaults to bf16 (fails on a CPU; on a T4 it would
    # pick emulated bf16). AlignTune imports DPOConfig from trl at call time, so a subclass fixes the precision at construction. train_dpo.py does the same.
    trl.DPOConfig = CpuDPOConfig  # module-level class: the Trainer pickles its args when it saves a checkpoint
    from aligntune.core.backend_factory import create_rl_trainer
    trainer = create_rl_trainer(
        model_name=str(merged), dataset_name=str(pairs), algorithm="dpo", backend="trl", output_dir=str(work / "out"),
        num_epochs=1, max_steps=2, batch_size=1, gradient_accumulation_steps=1, learning_rate=1e-5, max_seq_length=args.max_seq_length,
        use_peft=True, lora_r=8, lora_alpha=16, lora_target_modules="all-linear", beta=0.1, logging_steps=1, seed=42, precision="fp32",
        gradient_checkpointing=False, report_to="none",
    )
    trainer.train()
    hist = [h for h in trainer.trainer.state.log_history if "loss" in h]
    print("\nfirst logged steps:")
    for h in hist[:2]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k in ("step", "loss", "rewards/margins", "rewards/accuracies", "rewards/chosen", "rewards/rejected")})
    first = hist[0]
    ok_loss = abs(first["loss"] - math.log(2)) < 0.02
    ok_margin = abs(first.get("rewards/margins", 0.0)) < 0.02
    print(f"\nstep-1 loss {first['loss']:.4f} vs ln 2 = {math.log(2):.4f}: {'OK' if ok_loss else 'MISMATCH'};  reward margin {first.get('rewards/margins')}: {'OK' if ok_margin else 'MISMATCH'}")
    print("REFERENCE == POLICY AT START" if ok_loss and ok_margin else "REFERENCE IS NOT THE STARTING MODEL: do not train")


if __name__ == "__main__":
    main()
