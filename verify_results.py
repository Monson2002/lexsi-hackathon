"""Check an eval_results folder produced by eval.py before anyone trusts or quotes it.

    python verify_results.py res_all                       # run from the folder that has eval.py, test_b.jsonl, test_c.jsonl
    python verify_results.py res_all --adapter sft_v2_adapter/adapter
    python verify_results.py res_small --limit-ok          # a --limit smoke run: skip the document-count check

Checks per <system>_<split>.json (FAIL = do not use the result):
  1. the document count equals the dataset's (val 120, test 120, test_b 415, test_c 143)
  2. every metric RECOMPUTED from the raw outputs equals the stored headline (the table is honest), including the AuditKIT columns (ak_*) when present
  3. the model name carries the token cap and precision (@1280tok@bf16 / @fp16): not an old untagged cached run
  4. generation really happened (more than 0.3 s per document; a cached replay takes ~0 s)
  5. unfinished JSON outputs (start with { but never close): WARN, the cap may be too low
Across systems on the same split: identical outputs for two systems is a FAIL (an adapter that was not applied, or a cache replay).
With --adapter: the LoRA B matrices must not be all zero (an all-zero adapter behaves exactly like the base model).
"""
import argparse
import importlib.util
import json
import re
import statistics as st
import struct
import sys
from itertools import combinations
from pathlib import Path

SPLITS = ["test_a", "test_b", "test_c", "val", "test"]  # longest suffix first when parsing file names
EXPECTED = {"val": 120, "test": 120, "test_b": 415, "test_c": 143}
METRICS = ["parse_ok", "schema_valid", "field_f1", "doc_exact", "null_acc", "hallucination_rate", "hallucination_rate_adj"]
AK_METRICS = ["ak_json_schema_valid", "ak_field_f1", "ak_field_precision", "ak_field_recall"]


def load_eval():
    spec = importlib.util.spec_from_file_location("ev", "eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def lora_b_nonzero(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    base, total, nonzero = 8 + n, 0, 0
    for k, v in header.items():
        if k != "__metadata__" and "lora_B" in k:
            s, e = v["data_offsets"]
            f.seek(base + s)
            total += 1
            nonzero += any(f.read(e - s))
    return total, nonzero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--data-dir", default=".", help="folder with test_b.jsonl / test_c.jsonl")
    ap.add_argument("--dataset", default=None, help="HF dataset for val/test (default: the one eval.py uses)")
    ap.add_argument("--adapter", help="adapter folder to check for non-zero LoRA B weights")
    ap.add_argument("--limit-ok", action="store_true", help="results come from a --limit run")
    args = ap.parse_args()
    ev = load_eval()
    repo = args.dataset or "sujalgawas/aya-json-sft-dataset"
    fails, warns, outputs = [], [], {}

    def fail(msg):
        fails.append(msg)
        print("  FAIL", msg)

    def warn(msg):
        warns.append(msg)
        print("  WARN", msg)

    files = sorted(Path(args.results_dir).glob("*_*.json"))
    files = [p for p in files if p.name != "lexsi_provenance.json"]
    if not files:
        sys.exit(f"no result files in {args.results_dir}")
    data = {}
    for p in files:
        stem = p.stem
        split = next((s for s in SPLITS if stem.endswith("_" + s)), None)
        if split is None:
            continue
        system = stem[: -len(split) - 1]
        print(f"{system:10s} {split:7s}", end=" ")
        if split not in data:
            data[split] = {r["id"]: r for r in ev.load_split(split, args.data_dir, None, repo)}
        rows = data[split]
        j = json.load(open(p))
        P = j["predictions"]
        print(f"n={len(P)}")
        if len(P) != len(rows) and not args.limit_ok:
            fail(f"{p.name}: {len(P)} documents, dataset has {len(rows)}")
        missing = [x["sample_id"] for x in P if x["sample_id"] not in rows]
        if missing:
            fail(f"{p.name}: {len(missing)} ids are not in the dataset (wrong or updated dataset version?)")
            continue
        acc = {k: [] for k in METRICS}
        for x in P:
            r = rows[x["sample_id"]]
            m = ev.compute_field_metrics(x["raw_output"], r["gold"], r["schema"])
            m["hallucination_rate"] = ev.compute_hallucination_rate(x["raw_output"], r["document"])
            gold_strs = [v for v in ev.flatten_json_to_leaves(r["gold"]).values() if isinstance(v, str)]
            m["hallucination_rate_adj"] = ev.compute_hallucination_rate(x["raw_output"], r["document"], allowed=gold_strs)
            for k in METRICS:
                acc[k].append(m[k])
        if "ak_field_f1" in j["headline"]:  # AuditKIT's own metrics: recompute them with AuditKIT too
            ak_ms = ev.audit_kit_metrics()
            for k in AK_METRICS:
                acc[k] = []
            for x in P:
                sample = ev.to_samples([rows[x["sample_id"]]])[0]
                for m in ak_ms:
                    r = m.score(sample, x["raw_output"])
                    for sc in (r if isinstance(r, list) else [r]):
                        acc[sc.name].append(sc.value)
        worst = max(abs(st.mean(v) - j["headline"][k]) for k, v in acc.items())
        if worst > 1e-6:
            fail(f"{p.name}: recomputed metrics differ from the stored headline by {worst:.2e}")
        name = j["model_size"]["model_name"]
        if not re.search(r"@\d+tok@(bf16|fp16)$", name):
            fail(f"{p.name}: model name {name!r} has no token-cap/precision tag (old eval.py or an old cached run)")
        elif "@1280tok" not in name:
            warn(f"{p.name}: cap is not 1280 ({name})")
        secs = j["perf"]["latency_ms"]["mean"] / 1000
        if secs / max(len(P), 1) < 0.3:
            fail(f"{p.name}: {secs:.0f}s for {len(P)} documents looks like a cached replay, not generation")
        cut = sum(1 for x in P if (x["raw_output"] or "").strip().startswith("{") and not (x["raw_output"] or "").strip().endswith("}"))
        if cut:
            warn(f"{p.name}: {cut} outputs start a JSON object and never close it (cap too low, or the model did not stop)")
        outputs.setdefault(split, {})[system] = [x["raw_output"] for x in P]

    print()
    for split, by_sys in outputs.items():
        for a, b in combinations(by_sys, 2):
            if by_sys[a] == by_sys[b]:
                fail(f"{split}: systems {a} and {b} produced byte-identical outputs (adapter not applied, or cache replay)")
    if args.adapter:
        st_file = Path(args.adapter) / "adapter_model.safetensors"
        if st_file.exists():
            total, nz = lora_b_nonzero(st_file)
            print(f"adapter {args.adapter}: {nz} of {total} LoRA B matrices are non-zero")
            if nz == 0:
                fail("adapter is all zeros: it cannot change the model's output")
        else:
            fail(f"{st_file} not found")
    print(f"\n{len(files)} result files checked: {len(fails)} FAIL, {len(warns)} WARN")
    print("VERIFIED" if not fails else "NOT VERIFIED: do not quote these numbers")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
