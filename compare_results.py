"""Paired comparison of two systems from one eval.py run, plus the DPO success rule from docs/DPO_PLAN.md.

    python compare_results.py res_dpo sft_v2 sft_v2_dpo                  # A = reference, B = candidate; run from the folder with eval.py
    python compare_results.py res_all base sft_v2 --splits test_b test_c --no-rule

Per split and metric: mean of A, mean of B, the PAIRED difference B - A (same document under both systems) with a bootstrap 95% interval
(documents resampled with replacement). For doc_exact also an exact McNemar test on the documents where exactly one system is right.
Why paired: each document is easy or hard for both systems, so pairing removes most of the document-to-document noise. With 415 documents a single
proportion still has a standard error of about 2.4 points, so small gaps need this test before anyone calls them real.

The success rule (written before any result existed; docs/DPO_PLAN.md section 1), B is "better" only if ALL hold:
  test_b : doc_exact and field_f1 differences have a 95% interval above 0
  test_c : field_f1 difference has a 95% interval above 0 and doc_exact is not lower (doc_exact is near 0 on receipts, so it cannot decide)
  val, test : no metric drops significantly (interval entirely below 0) and parse_ok / schema_valid do not fall by more than 1 point
"""
import argparse
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

METRICS = ["parse_ok", "schema_valid", "field_f1", "doc_exact", "null_acc", "hallucination_rate_adj"]
HIGHER_IS_BETTER = {m: m != "hallucination_rate_adj" for m in METRICS}
SPLITS = ["test_a", "test_b", "test_c", "val", "test"]


def load_eval():
    spec = importlib.util.spec_from_file_location("ev", "eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def per_document(ev, rows, preds):
    """{sample_id: {metric: value}} recomputed from the raw outputs with the shared metric functions."""
    out = {}
    for p in preds:
        r = rows[p["sample_id"]]
        m = ev.compute_field_metrics(p["raw_output"], r["gold"], r["schema"])
        gold_strs = [v for v in ev.flatten_json_to_leaves(r["gold"]).values() if isinstance(v, str)]
        m["hallucination_rate_adj"] = ev.compute_hallucination_rate(p["raw_output"], r["document"], allowed=gold_strs)
        out[p["sample_id"]] = m
    return out


def mcnemar_exact(b, c):
    """Two-sided exact p-value for b documents only A gets right and c documents only B gets right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap(diff, n_boot, rng):
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("system_a", help="reference system, e.g. sft_v2")
    ap.add_argument("system_b", help="candidate system, e.g. sft_v2_dpo")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--dataset", default="sujalgawas/aya-json-sft-dataset")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rule", action="store_true", help="only print the table, do not apply the DPO success rule")
    args = ap.parse_args()
    ev = load_eval()
    rng = np.random.default_rng(args.seed)
    res = {}  # split -> metric -> (meanA, meanB, diff, lo, hi), plus McNemar
    for split in (args.splits or SPLITS):
        fa, fb = Path(args.results_dir) / f"{args.system_a}_{split}.json", Path(args.results_dir) / f"{args.system_b}_{split}.json"
        if not (fa.exists() and fb.exists()):
            continue
        rows = {r["id"]: r for r in ev.load_split(split, args.data_dir, None, args.dataset)}
        A = per_document(ev, rows, json.load(open(fa))["predictions"])
        B = per_document(ev, rows, json.load(open(fb))["predictions"])
        ids = sorted(set(A) & set(B))
        if len(ids) != len(A) or len(ids) != len(B):
            sys.exit(f"{split}: the two systems do not cover the same documents ({len(A)} vs {len(B)}, {len(ids)} in common)")
        print(f"\n=== {split}: {args.system_b} vs {args.system_a} (B - A), {len(ids)} paired documents ===")
        print(f"{'metric':24s}{'A':>8s}{'B':>8s}{'B-A':>9s}   95% interval")
        res[split] = {}
        for m in METRICS:
            a = np.array([A[i][m] for i in ids], dtype=float)
            b = np.array([B[i][m] for i in ids], dtype=float)
            d = b - a
            lo, hi = paired_bootstrap(d, args.boot, rng)
            res[split][m] = (a.mean(), b.mean(), d.mean(), lo, hi)
            print(f"{m:24s}{a.mean():8.3f}{b.mean():8.3f}{d.mean():+9.3f}   [{lo:+.3f}, {hi:+.3f}]" + ("   *" if lo > 0 or hi < 0 else ""))
        only_a = sum(1 for i in ids if A[i]["doc_exact"] == 1 and B[i]["doc_exact"] == 0)
        only_b = sum(1 for i in ids if A[i]["doc_exact"] == 0 and B[i]["doc_exact"] == 1)
        print(f"doc_exact: only A right {only_a}, only B right {only_b}, McNemar exact p = {mcnemar_exact(only_a, only_b):.4f}   (* = interval excludes 0)")
        res[split]["_mcnemar"] = (only_a, only_b)
    if not res:
        sys.exit("no split found for both systems")
    if args.no_rule:
        return
    print("\n=== DPO success rule (docs/DPO_PLAN.md section 1) ===")
    checks = []

    def need(split, label, ok):
        if split not in res:
            checks.append((f"{split}: {label}", None))
        else:
            checks.append((f"{split}: {label}", bool(ok)))

    g = lambda split, m: res[split][m]  # noqa: E731
    if "test_b" in res:
        need("test_b", "doc_exact gain, interval above 0", g("test_b", "doc_exact")[3] > 0)
        need("test_b", "field_f1 gain, interval above 0", g("test_b", "field_f1")[3] > 0)
    else:
        need("test_b", "doc_exact and field_f1 gain", False)
    if "test_c" in res:
        need("test_c", "field_f1 gain, interval above 0", g("test_c", "field_f1")[3] > 0)
        need("test_c", "doc_exact not lower", g("test_c", "doc_exact")[2] >= 0)
    else:
        need("test_c", "field_f1 gain", False)
    for split in ("val", "test"):
        if split in res:
            worst = [m for m in METRICS if (res[split][m][4] < 0 if HIGHER_IS_BETTER[m] else res[split][m][3] > 0)]
            need(split, f"no significant drop (found: {worst or 'none'})", not worst)
            need(split, "parse_ok and schema_valid fall by no more than 1 point", g(split, "parse_ok")[2] >= -0.01 and g(split, "schema_valid")[2] >= -0.01)
        else:
            need(split, "no significant drop", False)
    for label, ok in checks:
        print(f"  {'PASS' if ok else ('MISSING' if ok is None else 'FAIL'):8s}{label}")
    if all(ok for _, ok in checks):
        print(f"\nVERDICT: {args.system_b} helps (all conditions hold)")
    else:
        print(f"\nVERDICT: no reliable gain for {args.system_b}: report it as such and ship {args.system_a}")


if __name__ == "__main__":
    main()
