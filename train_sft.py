"""QLoRA SFT of Tiny Aya Global on data/sft_v2/sft_messages_train.jsonl, through AlignTune (TRL backend).

    python train_sft.py --dry-run                      # CPU, no model: shows exactly what the model would train on
    python train_sft.py --max-steps 5 --max-samples 64 # GPU smoke test (a few minutes)
    python train_sft.py                                # full run: 2 epochs

Needs: HF_TOKEN in the environment (CohereLabs/tiny-aya-global is gated), pip install aligntune==0.1.33 (pins trl 1.7.1,
peft, transformers 5.x; tested with transformers 5.20.2, torch 2.11, peft 0.21.2, bitsandbytes 0.50.2) and a CUDA GPU for the real run.

Things this script does on purpose (found by reading AlignTune 0.1.33 and testing on CPU, see the comments):
 1. task_type="sft", not "instruction_following". With "instruction_following" AlignTune turns our chat rows into plain
    prompt/completion STRINGS and TRL then trains without the chat template (no system turn, no turn tokens), which does not
    match how eval.py prompts the model. With "sft" it keeps system/user/assistant messages and TRL applies the template.
 2. Loss on the JSON answer only (AGENTS.md). TRL's assistant_only_loss needs {% generation %} markers in the chat template and
    the Cohere template has none; AlignTune only applies a custom template when the tokenizer has none. So after AlignTune loads
    the tokenizer we patch its own template: the assistant turn (answer + <|END_RESPONSE|> + <|END_OF_TURN_TOKEN|>) is wrapped
    in {% generation %}. The rendered text is byte-identical (checked on the training rows); only the mask changes.
 3. Rows longer than --max-seq-length (exact token count) are dropped up front: TRL would cut the END of a long row,
    i.e. the answer.
 4. Frozen prompt: the rows were written with the same system prompt and <JSON_SCHEMA>/<DOCUMENT> user message as eval.py.
"""
import argparse
import json
import os
import random
from pathlib import Path

SEED = 42
BASE = "CohereLabs/tiny-aya-global"
# Cohere template pieces we wrap (asserted below, so a different template fails loudly instead of silently training on everything)
ASSISTANT = "<|CHATBOT_TOKEN|><|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>{% elif role == 'system' %}"
EOT_TAIL = "{% endif %}<|END_OF_TURN_TOKEN|>{% endfor %}"


def default_template(tok):
    """The base tokenizer stores its template as a dict of named templates ({'default': ...}); the public copies are a plain string."""
    t = tok.chat_template
    if isinstance(t, dict):
        assert "default" in t, f"chat template names are {list(t)}: no 'default' to patch"
        return t["default"]
    return t


def patch_template(tok):
    t = default_template(tok)
    assert t.count(ASSISTANT) == 1 and t.count(EOT_TAIL) == 1, "chat template is not the expected Cohere one; do not train blind"
    tok.chat_template = t.replace(
        ASSISTANT, "<|CHATBOT_TOKEN|><|START_RESPONSE|>{% generation %}{{ message['content'] }}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>{% endgeneration %}{% elif role == 'system' %}"
    ).replace(EOT_TAIL, "{% endif %}{% if role != 'assistant' and role != 'chatbot' %}<|END_OF_TURN_TOKEN|>{% endif %}{% endfor %}")
    return t  # original default template (a string), for the identical-text check


def check_masks(tok, original, messages):
    """The patched template must render the same text, and the mask must cover exactly the answer + its two end tokens."""
    patched = tok.chat_template
    tok.chat_template = original
    plain = tok.apply_chat_template(messages, tokenize=False)
    tok.chat_template = patched
    assert tok.apply_chat_template(messages, tokenize=False) == plain, "patched template changed the rendered text"
    enc = tok.apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    sup = [i for i, m in zip(enc["input_ids"], enc["assistant_masks"]) if m]
    assert tok.decode(sup) == messages[-1]["content"] + "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>", "loss mask does not match the answer"
    return len(enc["input_ids"]), len(sup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v2/sft_messages_train.jsonl")
    ap.add_argument("--out", default="sft_v2_adapter")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1, help="per-device batch; the vocab is 262k so long rows are memory-hungry on a T4")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=2560)
    ap.add_argument("--max-samples", type=int, help="first N rows (smoke test)")
    ap.add_argument("--max-steps", type=int, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--model", default=BASE, help="base model; only change for a CPU smoke test with a tiny random model")
    ap.add_argument("--no-4bit", action="store_true", help="CPU smoke test only (bitsandbytes 4-bit needs CUDA)")
    ap.add_argument("--tokenizer-id", default=BASE, help="only for --dry-run without HF access, e.g. sujalgawas/aya-json-sft")
    ap.add_argument("--push-to", help="HF repo id to upload the adapter to (private). Not done unless you pass it")
    ap.add_argument("--dry-run", action="store_true", help="CPU only: load data through AlignTune, check the template and masks, build the training config, stop")
    args, unknown = ap.parse_known_args()
    if [u for i, u in enumerate(unknown) if u != "-f" and not (i and unknown[i - 1] == "-f")]:
        raise SystemExit(f"unknown arguments {unknown}")  # no silent ignoring (it cost us a run once)
    token = os.environ.get("HF_TOKEN")
    random.seed(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, token=token)
    original = patch_template(tok)

    # 3. drop rows whose exact token length exceeds the limit (before AlignTune sees them)
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.max_samples:
        rows = rows[: args.max_samples]
    kept = [r for r in rows if len(tok.apply_chat_template(r["messages"], tokenize=True, return_dict=True)["input_ids"]) <= args.max_seq_length]
    print(f"rows: {len(rows)} -> {len(kept)} after dropping {len(rows) - len(kept)} longer than {args.max_seq_length} tokens")
    data_path = out / "train_filtered.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_tok, n_sup = check_masks(tok, original, kept[0]["messages"])
    print(f"template check OK: identical text; first row {n_tok} tokens, {n_sup} supervised ({n_sup / n_tok:.0%}), all of them the answer")

    import torch
    # Ampere+ only: is_bf16_supported() is also True on a T4 (emulated bf16, ~several times slower than its native fp16)
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    from aligntune.core.backend_factory import create_sft_trainer
    trainer = create_sft_trainer(
        model_name=args.model, dataset_name=str(data_path), backend="trl", output_dir=str(out), task_type="sft",
        num_epochs=args.epochs, max_steps=args.max_steps,
        batch_size=args.batch, gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, max_seq_length=args.max_seq_length,
        use_peft=True, lora_r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, lora_target_modules="all-linear",
        quantization={} if args.no_4bit else {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
                                              "bnb_4bit_compute_dtype": "bfloat16" if bf16 else "float16"},
        precision="bf16" if bf16 else ("fp16" if torch.cuda.is_available() else "fp32"), bf16=bf16, fp16=torch.cuda.is_available() and not bf16, gradient_checkpointing=True,
        assistant_only_loss=True, warmup_ratio=0.05, weight_decay=0.01, seed=SEED, logging_steps=10, save_steps=100,
        load_best_model_at_end=False,
    )
    cfg = {k: v for k, v in vars(args).items()} | {"seed": SEED, "base": BASE, "bf16": bf16, "rows_used": len(kept)}
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2))
    print("config:", json.dumps(cfg))

    if args.dry_run:  # no model: only the data path and the training arguments
        trainer.tokenizer = tok
        trainer.setup_data()
        ds = trainer.train_dataset
        m = ds[0]["messages"]
        print("AlignTune dataset columns:", ds.column_names, "| rows:", len(ds))
        print("roles:", [x["role"] for x in m], "| system prompt intact:", m[0]["content"].startswith("You extract structured data."))
        from aligntune.core.sft.training_args_builder import build_sft_config
        opt = trainer._setup_optimizer_scheduler(dataset_for_estimation=ds)
        sc = build_sft_config(config=trainer.config, train_dataset=ds, eval_dataset=getattr(trainer, "eval_dataset", None), output_dir=str(out),
                              optim=opt["optimizer_name"], lr_scheduler_type=opt["scheduler_name"])
        for k in ("assistant_only_loss", "max_length", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "num_train_epochs",
                  "max_steps", "fp16", "bf16", "eval_strategy", "load_best_model_at_end", "gradient_checkpointing", "packing", "save_steps"):
            print(f"  TRL arg {k} = {getattr(sc, k, '<missing>')}")
        print("dry run finished; nothing was trained")
        return

    trainer.setup_model()  # AlignTune loads model (4-bit), applies LoRA, builds the tokenizer ...
    from collections import Counter
    print("trainable parameter dtypes after load:", dict(Counter(str(q.dtype) for q in trainer.model.parameters() if q.requires_grad)))
    if not bf16:
        # TRL 1.7.1's SFTTrainer.__init__ casts every trainable weight of a QUANTISED model to bf16, whatever precision we ask for
        # (trl/trainer/sft_trainer.py, "QLoRA ... converted to bf16"). With fp16=True the GradScaler then fails on bf16 gradients:
        # "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16" (seen on a Kaggle T4). So put the trainable
        # weights back to fp32 right before the optimizer is built. AlignTune imports SFTTrainer from trl at call time, so a subclass works.
        import trl

        class Fp32AdapterTrainer(trl.SFTTrainer):
            def create_optimizer(self, model=None):
                for q in (model if model is not None else self.model).parameters():
                    if q.requires_grad and q.dtype != torch.float32:
                        q.data = q.data.float()
                print("trainable dtypes at optimizer creation:", dict(Counter(str(q.dtype) for q in self.model.parameters() if q.requires_grad)))
                return super().create_optimizer(model)

        trl.SFTTrainer = Fp32AdapterTrainer
    if args.model != BASE:  # smoke test with a tiny model: borrow the Cohere template so the mask logic is exercised
        trainer.tokenizer.chat_template = original
    patch_template(trainer.tokenizer)  # ... and only now do we add the generation markers to ITS template
    result = trainer.train()
    print("training result:", result)
    if args.model != BASE:  # smoke test: show that TRL really masked everything except the answer
        d = trainer.trainer.train_dataset
        r = d[0]
        lab = r.get("labels") or r.get("completion_mask") or r.get("assistant_masks")
        print("TRL processed columns:", d.column_names)
        if "labels" in r:
            print(f"labels: {sum(x != -100 for x in r['labels'])} of {len(r['labels'])} tokens supervised ({sum(x != -100 for x in r['labels']) / len(r['labels']):.0%})")
    trainer.trainer.save_model(str(out / "adapter"))
    print(f"adapter saved to {out / 'adapter'}")
    if args.push_to:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.push_to, private=True, exist_ok=True)
        # no tokenizer files: the saved chat_template.jinja is OUR patched copy ({% generation %} tags that only transformers understands)
        api.upload_folder(folder_path=str(out / "adapter"), repo_id=args.push_to,
                          ignore_patterns=["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"])
        print("uploaded (private) to", args.push_to)


if __name__ == "__main__":
    main()
