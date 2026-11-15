"""Evaluate base / SFT / GRPO Tiny Aya on schema-conditioned extraction, via AuditKIT.

Run on a Kaggle GPU:   python eval.py --systems base sft grpo --splits val test_a
CPU smoke test:        python eval.py --stub --limit 10

Design (why it looks like this):
- The prompt format and the metric definitions are copied from Sujal's notebook (sections 5 and 6) so numbers
  match his. Do not edit them here without changing the notebook too (AGENTS.md: metrics defined once).
- AuditKIT runs the loop (ak.evaluate) and the comparison. Our extraction metrics are wrapped as one AuditKIT
  Metric, because AuditKIT's built-in FieldF1 scores nulls and hallucination differently from the training reward.
- A "system" is just (base model, optional adapter) in SYSTEMS, and a split is just a JSONL file. To benchmark
  something new, add a SYSTEMS entry or pass --data-dir with your own JSONL (same fields).
"""
import argparse
import dataclasses
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import auditkit as ak
import jsonschema
from auditkit.metric import Metric
from auditkit.metrics.structured import FieldF1, JsonSchemaValid
from auditkit.model import CallableModel
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction

SEED = 42
NUMERIC_REL_TOL = 1e-4
DATASET_REPO = "sujalgawas/tiny-aya-schema-extraction-testrun"
# The repo names are misleading. "tiny-aya-global-grpo-schema-adapter" is an OLD plain-SFT adapter whose lora_B weights
# are all zero (checked 2026-11-10), so it behaves exactly like base: kept as sft_old only for reference.
# "aya-json-sft" (2026-11-09) is the real SFT adapter: lora_B non-zero, base tiny-aya-global. It was trained on an
# earlier, smaller version of sujalgawas/aya-json-sft-dataset (300 train rows per its eval_summary.json), so evaluate
# it with --dataset sujalgawas/aya-json-sft-dataset --splits val test.
SYSTEMS = {
    "base": ("CohereLabs/tiny-aya-global", None),
    "sft": ("CohereLabs/tiny-aya-global", "sujalgawas/aya-json-sft"),
    "sft_old": ("CohereLabs/tiny-aya-global", "sujalgawas/tiny-aya-global-grpo-schema-adapter"),
    "grpo": ("CohereLabs/tiny-aya-global", "sujalgawas/tiny-aya-global-grpo-schema-model"),
}
GEN = {"max_new_tokens": 1280, "max_prompt_length": 2048, "batch_size": 8, "load_in_4bit": True}  # max_new_tokens differs from the notebook (384), see --max-new-tokens

# ---------------------------------------------------------------- prompt (frozen; copied from notebook section 5)
SYSTEM_PROMPT = (
    "You extract structured data. Return only a single JSON object that conforms to the given JSON Schema. "
    "Use null for values that are not present in the document. Do not invent values. "
    "Do not add explanations or markdown fences."
)


def build_user_message(schema_dict, document_text):
    schema_str = json.dumps(schema_dict, ensure_ascii=False, indent=2)
    return f"<JSON_SCHEMA>\n{schema_str}\n</JSON_SCHEMA>\n\n<DOCUMENT>\n{document_text}\n</DOCUMENT>"


# ---------------------------------------------------------------- metrics (copied from notebook section 6)
def normalize_for_grounding(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_optional_code_fence(text):
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_json_object(text):
    cleaned = strip_optional_code_fence(text)
    if not cleaned:
        return False, None
    try:
        obj = json.loads(cleaned)
        return (True, obj) if isinstance(obj, dict) else (False, None)
    except Exception:
        return False, None


def check_schema_valid(text, schema):
    ok, obj = parse_json_object(text)
    if not ok:
        return False
    try:
        jsonschema.validate(instance=obj, schema=schema)
        return True
    except Exception:
        return False


def flatten_json_to_leaves(obj, prefix=""):
    leaves = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)) and len(v) > 0:
                leaves.update(flatten_json_to_leaves(v, path))
            elif isinstance(v, (dict, list)):
                leaves[path] = None
            else:
                leaves[path] = v
    elif isinstance(obj, list):
        for idx, elem in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(elem, (dict, list)) and len(elem) > 0:
                leaves.update(flatten_json_to_leaves(elem, path))
            else:
                leaves[path] = elem
    return leaves


def normalize_date_string(val_str):
    s = val_str.strip()
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return s.lower()


def leaves_match(gold_val, pred_val, rel_tol=NUMERIC_REL_TOL):
    if gold_val is None and pred_val is None:
        return True
    if (gold_val is None) != (pred_val is None):
        return False
    if isinstance(gold_val, bool) or isinstance(pred_val, bool):
        return type(gold_val) is bool and type(pred_val) is bool and gold_val == pred_val
    if isinstance(gold_val, (int, float)) and isinstance(pred_val, (int, float)):
        denom = max(abs(float(gold_val)), 1e-9)
        return abs(float(gold_val) - float(pred_val)) / denom <= rel_tol
    gs = normalize_date_string(re.sub(r"\s+", " ", str(gold_val).strip()))
    ps = normalize_date_string(re.sub(r"\s+", " ", str(pred_val).strip()))
    return gs == ps


def compute_field_metrics(text, gold, schema):
    ok, pred = parse_json_object(text)
    gold_leaves = flatten_json_to_leaves(gold)
    gold_null = [p for p, v in gold_leaves.items() if v is None]
    if not ok or pred is None:
        return {"parse_ok": 0.0, "schema_valid": 0.0, "field_f1": 0.0, "doc_exact": 0.0, "null_acc": 0.0}
    s_valid = 1.0 if check_schema_valid(text, schema) else 0.0
    pred_leaves = flatten_json_to_leaves(pred)
    tp = fp = fn = null_correct = 0
    for path in set(gold_leaves) | set(pred_leaves):
        in_gold = path in gold_leaves
        g_val, p_val = gold_leaves.get(path), pred_leaves.get(path)
        if in_gold:
            if g_val is None:
                if p_val is None:
                    tp += 1
                    null_correct += 1
                else:
                    fp += 1
            elif p_val is None:
                fn += 1
            elif leaves_match(g_val, p_val):
                tp += 1
            else:
                fp += 1
                fn += 1
        elif p_val is not None:
            fp += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "parse_ok": 1.0, "schema_valid": s_valid, "field_f1": f1,
        "doc_exact": 1.0 if (fp == 0 and fn == 0 and s_valid == 1.0) else 0.0,
        "null_acc": null_correct / len(gold_null) if gold_null else 1.0,
    }


def compute_hallucination_rate(text, document, allowed=()):
    """Notebook definition when `allowed` is empty. `allowed` = values that may be absent from the document without
    counting as invented (used for hallucination_rate_adj, see ExtractionMetrics)."""
    ok, pred = parse_json_object(text)
    if not ok or pred is None:
        return 0.0
    norm_doc = normalize_for_grounding(document)
    ok_vals = {normalize_for_grounding(a) for a in allowed}
    vals = [v for v in flatten_json_to_leaves(pred).values() if isinstance(v, str) and v.strip()]
    if not vals:
        return 0.0
    return sum(1 for v in vals if normalize_for_grounding(v) and normalize_for_grounding(v) not in norm_doc
               and normalize_for_grounding(v) not in ok_vals) / len(vals)


class ExtractionMetrics(Metric):
    """All six extraction metrics as one AuditKIT metric. Sample.metadata carries schema, document and gold."""

    name = "extraction"
    direction = Direction.MAXIMIZE  # hallucination_rate is lower-is-better; its Score overrides direction below

    def score(self, sample, output, context=None):
        m = sample.metadata
        out = compute_field_metrics(output, m["gold"], m["schema"])
        out["hallucination_rate"] = compute_hallucination_rate(output, m["document"])
        # Extra, not in the notebook: FieldBench gold is normalized (dates as YYYY-MM-DD, enum labels, currency codes),
        # so even a perfect answer is "not in the document" verbatim and scores 0.3-0.5 on the original metric. This
        # variant forgives a value that equals a gold string, so only values that are neither grounded nor expected
        # count. On Sujal's data (gold is extractive) both columns agree.
        gold_strs = [v for v in flatten_json_to_leaves(m["gold"]).values() if isinstance(v, str)]
        out["hallucination_rate_adj"] = compute_hallucination_rate(output, m["document"], allowed=gold_strs)
        return [
            Score(name=k, value=v, direction=Direction.MINIMIZE if k.startswith("hallucination") else Direction.MAXIMIZE)
            for k, v in out.items()
        ]


class AuditKitMetric(Metric):
    """Runs one of AuditKIT's own metrics and prefixes its score names with 'ak_'.

    AuditKIT's FieldF1 emits a score called 'field_f1', which is also the name of OUR column (Sujal's definition, which the training reward
    uses). Two metrics may not share a score name in one run, and the two definitions differ (null handling, relative vs absolute number
    tolerance, list alignment), so the AuditKIT versions are reported next to ours as ak_field_f1 / ak_field_precision / ak_field_recall /
    ak_json_schema_valid and never replace them.
    """

    direction = Direction.MAXIMIZE  # AuditKIT requires it on the class; all four wrapped metrics are higher-is-better

    def __init__(self, inner):
        assert inner.direction == Direction.MAXIMIZE
        self.inner, self.name, self.kind, self.required_fields = inner, "ak_" + inner.name, inner.kind, inner.required_fields

    def identity(self):
        return {**self.inner.identity(), "name": self.name}

    def applicable(self, sample):
        return self.inner.applicable(sample)

    def score(self, sample, output, context=None):
        r = self.inner.score(sample, output, context)
        return [dataclasses.replace(x, name="ak_" + x.name) for x in (r if isinstance(r, list) else [r])]


def audit_kit_metrics():
    """AuditKIT built-ins for the same task. strip_fences=True because our own parser also accepts one ```json fence."""
    return [
        AuditKitMetric(JsonSchemaValid(strip_fences=True)),  # schema from sample.metadata["schema"]
        AuditKitMetric(FieldF1(normalize_numbers=True, number_tol=1e-4, case_sensitive=False, strip_fences=True)),  # target = the gold JSON
    ]


# ---------------------------------------------------------------- data
def load_split(split, data_dir=None, limit=None, repo=DATASET_REPO):
    """JSONL with id, domain, schema, document, gold. From --data-dir if <split>.jsonl exists there, else the public
    HF dataset. So `--data-dir fieldbench_out --splits test_b test_c val` mixes FieldBench and Sujal's splits."""
    if data_dir and (Path(data_dir) / f"{split}.jsonl").exists():
        path = Path(data_dir) / f"{split}.jsonl"
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, f"{split}.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows[:limit] if limit else rows


def to_samples(rows):
    # Sample.input is the user message; the model callable adds the system prompt and chat template.
    return [
        Sample(
            id=r["id"], input=build_user_message(r["schema"], r["document"]), target=json.dumps(r["gold"]),
            tags=[r["domain"]], metadata={"schema": r["schema"], "document": r["document"], "gold": r["gold"]},
        )
        for r in rows
    ]


# ---------------------------------------------------------------- models
def stub_model(prompts):
    """CPU smoke test: echoes the gold object, so every metric should be perfect."""
    out = []
    for p in prompts:
        out.append(json.dumps(STUB_GOLD[p]))
    return out


def native_bf16():
    """True only on Ampere or newer. torch.cuda.is_bf16_supported() is also True on a T4/P100 (emulated, several times slower),
    so Kaggle T4 runs were ~10x slower than a native-bf16 GPU. Older GPUs use fp16, which they run natively."""
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def hf_model(base_id, adapter_id):
    """Returns a list[str] -> list[str] callable. Handles a LoRA adapter repo or a full/merged model repo."""
    import torch
    from huggingface_hub import list_repo_files
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(base_id, token=token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation needs left padding
    tok.truncation_side = "left"  # if too long, drop the start, never the generation prompt
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if native_bf16() else torch.float16
    ) if GEN["load_in_4bit"] else None
    if adapter_id and os.path.isdir(adapter_id):  # a local folder: an adapter if it has adapter_config.json, else a full (merged) model. No Hub lookup for paths
        is_adapter = os.path.exists(os.path.join(adapter_id, "adapter_config.json"))
    else:
        is_adapter = adapter_id and "adapter_config.json" in list_repo_files(adapter_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        base_id if (is_adapter or not adapter_id) else adapter_id,
        token=token, quantization_config=bnb, device_map="auto", trust_remote_code=True,
        **({} if native_bf16() else {"dtype": torch.float16}),  # non-quantised layers too, or they stay bf16 (emulated) on a T4
    )
    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()

    def run(prompts):
        outs = []
        for i in range(0, len(prompts), GEN["batch_size"]):
            texts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
                for p in prompts[i:i + GEN["batch_size"]]
            ]
            add_special = not (tok.bos_token and texts[0].startswith(tok.bos_token))  # template already adds BOS
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=GEN["max_prompt_length"], add_special_tokens=add_special).to(model.device)
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=GEN["max_new_tokens"], do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            outs += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in ids]
        return outs

    return run


# ---------------------------------------------------------------- main
def main():
    global STUB_GOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS),
                    help=f"{list(SYSTEMS)} or name=<adapter folder or HF repo> to evaluate a new adapter on {SYSTEMS['base'][0]}, e.g. sft_v2=sft_v2_adapter/adapter; "
                         f"name=<adapter>@<base folder> puts the adapter on another base; name=<full model folder> alone evaluates a merged model")
    ap.add_argument("--splits", nargs="+", default=["val", "test_a"], help="val test_a (Sujal), test_b test_c (FieldBench, via fieldbench_to_jsonl.py + --data-dir)")
    ap.add_argument("--data-dir", help="local folder with <split>.jsonl instead of the HF dataset")
    ap.add_argument("--dataset", default=DATASET_REPO, help="HF dataset repo with <split>.jsonl (fields: id, domain, schema, document, gold)")
    ap.add_argument("--max-new-tokens", type=int, default=GEN["max_new_tokens"],
                    help="generation cap. Default 1280: at 384 (the old notebook value) FieldBench answers of ~900 tokens were cut off; at 1280, 1536 and 2048 "
                         "base/GRPO outputs were byte-identical, so no output needs more")
    ap.add_argument("--limit", type=int, help="first N examples per split (smoke tests)")
    ap.add_argument("--stub", action="store_true", help="no GPU: model echoes gold; checks the pipeline only")
    ap.add_argument("--out", default="eval_results")
    args, unknown = ap.parse_known_args()  # tolerate Jupyter/Colab injecting "-f kernel.json" if run inside a cell
    unknown = [u for i, u in enumerate(unknown) if u != "-f" and not (i and unknown[i - 1] == "-f")]
    if unknown:  # an old eval.py silently ignored --max-new-tokens and replayed cached 384-token results
        raise SystemExit(f"unknown arguments {unknown}: is this the latest eval.py? (python eval.py -h)")

    GEN["max_new_tokens"] = args.max_new_tokens
    Path(args.out).mkdir(exist_ok=True)
    for i, name in enumerate(args.systems):  # 'name=adapter' registers a new system on the fly
        if "=" in name:
            name, adapter = name.split("=", 1)
            base = SYSTEMS["base"][0]
            if "@" in adapter:  # name=adapter@base: the adapter sits on another base (e.g. a DPO adapter on the merged SFT model)
                adapter, base = adapter.split("@", 1)
            SYSTEMS[name] = (base, adapter)
            args.systems[i] = name
        elif name not in SYSTEMS:
            raise SystemExit(f"unknown system {name!r}; use one of {list(SYSTEMS)} or name=<adapter>")
    systems = ["stub"] if args.stub else args.systems
    results = {}
    for name in systems:
        model = None
        for split in args.splits:
            samples = to_samples(load_split(split, args.data_dir, args.limit, args.dataset))
            if args.stub:
                STUB_GOLD = {s.input: s.metadata["gold"] for s in samples}
                model = CallableModel(stub_model, name="stub")
            elif model is None:
                # Explicit name: AuditKIT caches runs by a fingerprint of the model callable. hf_model() returns the
                # same function body for every system, so without a unique name sft/grpo silently replay base's
                # cached outputs (this happened in the first Kaggle run).
                # The cache key does not see GEN, so the token cap goes into the name, or a rerun with another cap would replay
                # the cached outputs of the old one.
                tag = f"@{args.max_new_tokens}tok@{'bf16' if native_bf16() else 'fp16'}"  # dtype too: fp16 and bf16 outputs differ slightly; always: untagged names are old 384-token caches and must never match
                model = CallableModel(hf_model(*SYSTEMS[name]), name=f"{name}:{SYSTEMS[name][0]}+{SYSTEMS[name][1]}{tag}")
            t0 = time.time()
            res = ak.evaluate(samples, model, [ExtractionMetrics(), *audit_kit_metrics()], experiment_name=f"{name}-{split}")
            results[(name, split)] = res
            res.save(str(Path(args.out) / f"{name}_{split}.json"))
            print(f"{name:5s} {split:7s} {time.time() - t0:6.0f}s", {k: round(v, 3) for k, v in res.headline.items()})

    # one markdown table: rows = system/split, columns = metrics
    cols = ["parse_ok", "schema_valid", "field_f1", "doc_exact", "null_acc", "hallucination_rate", "hallucination_rate_adj"]
    ak_cols = ["ak_json_schema_valid", "ak_field_f1", "ak_field_precision", "ak_field_recall"]  # AuditKIT's own metrics, see AuditKitMetric
    lines = ["| system | split | " + " | ".join(cols + ak_cols) + " | docs/s | s per 1000 docs |", "|" + "---|" * (len(cols) + len(ak_cols) + 4)]
    for (name, split), res in results.items():
        rps = (res.perf or {}).get("throughput", {}).get("rps") or float("nan")  # AuditKIT's own throughput record for the run
        lines.append(f"| {name} | {split} | " + " | ".join(f"{res.headline.get(c, float('nan')):.3f}" for c in cols + ak_cols)
                     + f" | {rps:.2f} | {1000 / rps if rps == rps and rps else float('nan'):.0f} |")
    (Path(args.out) / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
