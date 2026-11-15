"""Sample the SFT v2 model on the fresh DPO prompts, to collect its own wrong answers (docs/DPO_PLAN.md WP2b). Needs a GPU.

    python sample_onpolicy.py --adapter monson2002/tiny-aya-sft-v2-adapter --adapter-revision 9774195260373ce418e3e3869cd3c7bef84f47b8 \\
        --pool pool_clean.jsonl --out onpolicy_samples.jsonl --limit 16        # smoke test first; --adapter also accepts a local adapter folder
    (same command without --limit for all prompts)

Output: one JSON line per sample {id, sample_index, raw_output, temperature}. RESUMABLE: ids already in --out are skipped, so after an interruption just run
the same command again. Prompts are the frozen prompt (same system prompt and user message as eval.py); decoding is SAMPLING (temperature 0.7 by default),
because greedy decoding gives one answer per prompt and hides the model's variability. The model is loaded exactly like eval.py (4 bit, fp16 below Ampere).
Gold is NOT read here: this script never sees the answers.
The adapter can be a Hugging Face repo id. Pin its commit with --adapter-revision: the script resolves it, checks it, and writes <out>.meta.json (adapter, commit, weights SHA-256,
base, sampling settings), so the samples can always be traced to the exact weights that produced them.
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path


def load_eval():
    spec = importlib.util.spec_from_file_location("ev", str(Path(__file__).resolve().parent / "eval.py")
                                                  if (Path(__file__).resolve().parent / "eval.py").exists() else "eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def resolve_adapter(args, token):
    """Provenance of the adapter: for a Hub repo, the exact commit and the weights hash. A pinned revision that is not found stops the run."""
    meta = {"adapter": args.adapter, "adapter_revision_requested": args.adapter_revision}
    if os.path.isdir(args.adapter):
        meta["adapter_source"] = "local folder"
        return meta
    from huggingface_hub import HfApi
    info = HfApi().model_info(args.adapter, revision=args.adapter_revision, files_metadata=True, token=token)
    meta["adapter_source"] = "huggingface hub"
    meta["adapter_resolved_commit"] = info.sha
    meta["adapter_weights_sha256"] = next((x.lfs["sha256"] for x in info.siblings if x.rfilename == "adapter_model.safetensors" and x.lfs), None)
    if args.adapter_revision and info.sha != args.adapter_revision:
        raise SystemExit(f"adapter commit {info.sha} is not the pinned {args.adapter_revision}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="SFT v2 adapter: a local folder or a Hugging Face repo id")
    ap.add_argument("--adapter-revision", help="commit of the adapter repo to use (pin it; recorded in <out>.meta.json)")
    ap.add_argument("--pool", required=True, help="pool_clean.jsonl (id, schema, document, gold)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--n", type=int, default=1, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1280)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N prompts only (smoke test)")
    ap.add_argument("--stub", action="store_true", help="no GPU: answers are made up from the pool's gold, to test the pipeline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, help="torch CPU threads (only matters if running without a GPU)")
    args, unknown = ap.parse_known_args()
    if unknown:
        raise SystemExit(f"unknown arguments {unknown}")
    ev = load_eval()
    rows = [json.loads(line) for line in open(args.pool, encoding="utf-8")]
    import random as _random
    _random.Random(args.seed).shuffle(rows)  # the pool is ordered by source schema set: shuffle so that a run stopped early still covers a random subset
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    # Speed: within each block of 256 prompts (still in shuffled order, so an early stop stays a random sample) sort by length, so a batch has
    # similar-length prompts and wastes little time on padding.
    todo = [r for i in range(0, len(todo), 256) for r in sorted(todo[i:i + 256], key=lambda r: len(r["document"]) + len(json.dumps(r["schema"])))]
    print(f"{len(rows)} prompts, {len(done)} already sampled, {len(todo)} to do", flush=True)
    if not todo:
        return
    token = os.environ.get("HF_TOKEN")
    meta = {"base": args.base, "temperature": args.temperature, "top_p": args.top_p, "n": args.n, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
            "pool": os.path.basename(args.pool), "stub": args.stub}
    meta.update({"adapter": args.adapter, "adapter_source": "stub"} if args.stub else resolve_adapter(args, token))
    Path(str(args.out) + ".meta.json").parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print("provenance:", json.dumps({k: meta[k] for k in meta if k.startswith("adapter")}), flush=True)

    if args.stub:
        import random
        rng = random.Random(args.seed)

        def generate(prompts, rows_batch):
            res = []
            for r in rows_batch:
                g = json.loads(json.dumps(r["gold"]))
                if rng.random() < 0.5 and g:
                    g.pop(next(iter(g)))  # half of the answers lose a key
                res.append(json.dumps(g, ensure_ascii=False, sort_keys=True))
            return res
    else:
        import torch
        if args.threads:
            torch.set_num_threads(args.threads)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(args.base, token=token, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side, tok.truncation_side = "left", "left"
        bf16 = ev.native_bf16()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.base, token=token, quantization_config=bnb, device_map="auto", trust_remote_code=True,
                                                     **({} if bf16 else {"dtype": torch.float16}))
        model = PeftModel.from_pretrained(model, args.adapter, **({"revision": args.adapter_revision} if args.adapter_revision and not os.path.isdir(args.adapter) else {}))
        model.eval()
        torch.manual_seed(args.seed)

        def generate(prompts, rows_batch):
            texts = [tok.apply_chat_template([{"role": "system", "content": ev.SYSTEM_PROMPT}, {"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True) for p in prompts]
            add_special = not (tok.bos_token and texts[0].startswith(tok.bos_token))
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=ev.GEN["max_prompt_length"],
                      add_special_tokens=add_special).to(model.device)
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                     pad_token_id=tok.pad_token_id)
            return [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in ids]

    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n_done = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            prompts = [ev.build_user_message(r["schema"], r["document"]) for r in batch]
            for k in range(args.n):
                for r, raw in zip(batch, generate(prompts, batch)):
                    f.write(json.dumps({"id": r["id"], "sample_index": k, "raw_output": raw, "temperature": args.temperature}, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(batch)
            if (i // args.batch_size) % 5 == 0:
                rate = n_done / (time.time() - t0)
                print(f"{n_done}/{len(todo)} prompts, {rate:.2f} prompts/s, eta {(len(todo) - n_done) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    print(f"done: {n_done} prompts in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
