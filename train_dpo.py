"""DPO on top of SFT v2, through AlignTune (TRL backend). docs/DPO_PLAN.md WP5.

    python merge_sft_adapter.py --adapter sft_v2_adapter/adapter --out merged_sft_v2      # first: SFT v2 merged into the weights (reference = SFT v2)
    python train_dpo.py --model merged_sft_v2 --dry-run                                    # CPU, no model: data + config checks
    python train_dpo.py --model merged_sft_v2 --max-samples 32 --max-steps 3               # GPU smoke test
    python train_dpo.py --model merged_sft_v2                                              # full run

Needs HF_TOKEN, pip install aligntune==0.1.33 (same pins as the SFT run), and a CUDA GPU for a real run.

What this does on purpose (each found by reading AlignTune 0.1.33 / TRL 1.7.1, and testing on CPU):
 1. Policy and reference both start as SFT v2: the model is the MERGED SFT checkpoint and a fresh LoRA goes on top; AlignTune's DPO uses "adapter off" as the
    reference. Verified on a tiny model: first-step loss 0.6931 = ln 2 and reward margin 0 (data/dpo_start_check.py). The GPU smoke test re-checks it: the
    first logged loss must be close to 0.693, otherwise do not train.
 2. TRL's DPOConfig defaults to max_length=1024 with truncation keep_start: our prompts alone are about 1200 tokens, so the answers (the part DPO compares) would be
    cut off. max_length is set explicitly, and pairs that do not fit are dropped up front (counted).
 3. TRL's DPOConfig also defaults bf16 on; on a T4 that means emulated bf16, several times slower, and a CPU refuses it. Precision is fixed explicitly:
    bf16 on Ampere or newer, fp16 below.
 4. TRL's DPOTrainer casts every trainable weight of a quantised model to bf16 in its constructor (same as SFTTrainer did): with fp16 the GradScaler then crashes.
    On pre-Ampere GPUs the trainable weights are put back to fp32 right before the optimizer is built.
 5. AlignTune's data loader flattens chat-format columns (message lists) into plain strings before TRL sees them: the system prompt ends up glued to the user text and
    the chat template is never applied, so DPO would train on a different prompt format than eval.py uses. So the chat template is rendered HERE into plain strings
    that survive the flattening: prompt = template(system, user) with the generation prompt, chosen/rejected = the answer + <|END_RESPONSE|><|END_OF_TURN_TOKEN|>.
    The leading BOS is stripped from the prompt because TRL's string path tokenizes with default special tokens and adds one itself (otherwise: two BOS).
    Checked on the real tokenizer: the token ids TRL will build equal the chat-template ids for 600 of 600 chosen/rejected sequences, and the run stops if that check fails.
 AlignTune imports DPOConfig/DPOTrainer from trl when it builds the trainer, so module-level subclasses (picklable, the Trainer saves its args) are swapped in.
"""
import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch
import trl

SEED = 42
PREC = {"bf16": False, "fp16": False, "use_cpu": False, "max_length": 2048}  # filled in main() before AlignTune builds the trainer


class FixedDPOConfig(trl.DPOConfig):
    def __post_init__(self):
        self.bf16, self.fp16, self.use_cpu = PREC["bf16"], PREC["fp16"], PREC["use_cpu"]
        self.max_length, self.truncation_mode = PREC["max_length"], "keep_start"
        super().__post_init__()


class Fp32AdapterDPOTrainer(trl.DPOTrainer):
    def create_optimizer(self, model=None):
        if PREC["fp16"]:
            for q in (model if model is not None else self.model).parameters():
                if q.requires_grad and q.dtype != torch.float32:
                    q.data = q.data.float()
            print("trainable dtypes at optimizer creation:", dict(Counter(str(q.dtype) for q in self.model.parameters() if q.requires_grad)))
        return super().create_optimizer(model)


def render_rows(tok, rows, check=50):
    """Message-list pairs -> plain strings that tokenize (TRL string path) exactly like the chat template. Raises if they do not."""
    adds_bos = bool(tok.bos_token) and tok("x")["input_ids"][0] == tok.bos_token_id
    out = []
    for r in rows:
        p = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        item = {"kind": r.get("kind")}
        for which in ("chosen", "rejected"):
            full = tok.apply_chat_template(r["prompt"] + r[which], tokenize=False)
            assert full.startswith(p), "chat template: the prompt is not a prefix of the full conversation"
            item[which] = full[len(p):]
        item["prompt"] = p[len(tok.bos_token):] if adds_bos and p.startswith(tok.bos_token) else p
        out.append(item)
    for r, o in list(zip(rows, out))[:check]:
        for which in ("chosen", "rejected"):
            ref = tok.apply_chat_template(r["prompt"] + r[which], tokenize=True, return_dict=True)["input_ids"]
            mine = tok(text=o["prompt"] + o[which])["input_ids"]
            if ref != mine:
                raise SystemExit("rendered DPO strings do not tokenize like the chat template (BOS/EOS mismatch): do not train, the prompt format would differ from eval")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged SFT v2 folder (merge_sft_adapter.py)")
    ap.add_argument("--pairs", default="data/dpo/dpo_pairs_train.jsonl")
    ap.add_argument("--out", default="dpo_adapter")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5, help="UNTUNED starting point for a LoRA; tune only if time allows, and never on val/test/FieldBench")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta (AlignTune/TRL default)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048, help="max tokens of prompt + answer; longer pairs are dropped")
    ap.add_argument("--max-samples", type=int, help="first N pairs (smoke test)")
    ap.add_argument("--max-steps", type=int, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--no-4bit", action="store_true", help="CPU smoke test only")
    ap.add_argument("--push-to", help="HF repo id to upload the adapter to (private); not done unless given")
    ap.add_argument("--dry-run", action="store_true")
    args, unknown = ap.parse_known_args()
    if [u for i, u in enumerate(unknown) if u != "-f" and not (i and unknown[i - 1] == "-f")]:
        raise SystemExit(f"unknown arguments {unknown}")
    token = os.environ.get("HF_TOKEN")
    random.seed(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8  # not is_bf16_supported(): that is also True on a T4 (emulated)
    PREC.update(bf16=bf16, fp16=cuda and not bf16, use_cpu=not cuda, max_length=args.max_length)
    trl.DPOConfig, trl.DPOTrainer = FixedDPOConfig, Fp32AdapterDPOTrainer

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    rows = [json.loads(line) for line in open(args.pairs, encoding="utf-8")]
    if args.max_samples:
        rows = rows[: args.max_samples]

    def n_tokens(r, which):
        return len(tok.apply_chat_template(r["prompt"] + r[which], tokenize=True, return_dict=True)["input_ids"])

    kept = [r for r in rows if max(n_tokens(r, "chosen"), n_tokens(r, "rejected")) <= args.max_length]
    print(f"pairs: {len(rows)} -> {len(kept)} after dropping {len(rows) - len(kept)} longer than {args.max_length} tokens (prompt + answer); kinds: {dict(Counter(r['kind'] for r in kept))}")
    rendered = render_rows(tok, kept)
    print(f"rendered {len(rendered)} pairs to plain strings; token-level check against the chat template passed")
    data = out / "pairs_filtered.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for r in rendered:
            f.write(json.dumps({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}, ensure_ascii=False) + "\n")

    from aligntune.core.backend_factory import create_rl_trainer
    trainer = create_rl_trainer(
        model_name=args.model, dataset_name=str(data), algorithm="dpo", backend="trl", output_dir=str(out),
        num_epochs=args.epochs, max_steps=args.max_steps if args.max_steps else -1, batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, max_seq_length=args.max_length, use_peft=True, lora_r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        lora_target_modules="all-linear", beta=args.beta, loss_type="sigmoid", logging_steps=1, save_steps=100, seed=SEED,
        quantization={} if args.no_4bit else {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
                                              "bnb_4bit_compute_dtype": "bfloat16" if bf16 else "float16"},
        precision="bf16" if bf16 else ("fp16" if cuda else "fp32"), gradient_checkpointing=True, report_to="none",
    )
    cfg = vars(args) | {"seed": SEED, "bf16": bf16, "fp16": PREC["fp16"], "pairs_used": len(kept)}
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2))
    print("config:", json.dumps(cfg))

    if args.dry_run:
        trainer.tokenizer = tok
        trainer.setup_data()
        ds = trainer.train_dataset
        r = ds[0]
        print("AlignTune dataset columns:", ds.column_names, "| rows:", len(ds))
        print("types:", {k: type(r[k]).__name__ for k in ("prompt", "chosen", "rejected")}, "| prompt starts:", repr(r["prompt"][:90]))
        print("prompt ends:", repr(r["prompt"][-40:]), "| chosen ends:", repr(r["chosen"][-45:]), "| system prompt intact:", "You extract structured data." in r["prompt"])
        c = FixedDPOConfig(output_dir=str(out / "cfg_check"), per_device_train_batch_size=args.batch, beta=args.beta, loss_type=["sigmoid"], learning_rate=args.lr)
        for k in ("max_length", "truncation_mode", "bf16", "fp16", "use_cpu", "beta", "gradient_checkpointing", "learning_rate"):
            print(f"  TRL arg {k} = {getattr(c, k)}")
        print("dry run finished; nothing was trained")
        return

    result = trainer.train()
    print("training result:", result)
    hist = [h for h in trainer.trainer.state.log_history if "loss" in h]
    if hist:
        first = hist[0]
        print("first logged step:", {k: round(v, 4) if isinstance(v, float) else v for k, v in first.items() if k.startswith("rewards") or k in ("loss", "step")})
        print("last logged step :", {k: round(v, 4) if isinstance(v, float) else v for k, v in hist[-1].items() if k.startswith("rewards") or k in ("loss", "step")})
        if abs(first["loss"] - 0.6931) > 0.05:
            print("WARNING: the first-step loss is not ln 2 (0.693): policy and reference did NOT start equal. Do not use this run.")
    trainer.trainer.save_model(str(out / "adapter"))
    print(f"adapter saved to {out / 'adapter'}")
    if args.push_to:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.push_to, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out / "adapter"), repo_id=args.push_to, ignore_patterns=["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"])
        print("uploaded (private) to", args.push_to)


if __name__ == "__main__":
    main()
