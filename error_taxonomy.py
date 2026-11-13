"""Count WHICH kinds of mistakes each system makes, from the raw outputs of one eval.py run.

    python error_taxonomy.py res_all                        # every system and split in the folder; run where eval.py is
    python error_taxonomy.py res_all --systems sft_v2 --splits test_b test_c --json taxonomy.json

Document-level flags (a document can have several):
  unparseable_unfinished  starts a JSON object and never closes it (cut off, or the model did not stop)
  unparseable_other       not valid JSON for another reason
  fenced                  the answer is wrapped in a ``` fence (accepted by our parser, but the prompt forbids it)
  schema_invalid          valid JSON that fails the schema; its first error type is counted as schema_invalid:<required|additionalProperties|enum|type|other>
  schema_echo             schema vocabulary (type / required / properties / $schema ...) appears as keys that the gold does not have
Leaf-level counts (a leaf is one value at one path; computed on answers that parse):
  missing_key        the gold has a value at this path and the answer has no such key
  extra_key          the answer has a non-null value at a path the gold does not have
  date_unnormalised  gold is an ISO date, the answer is a date written another way
  invented_value     gold is null, the answer has a value
  over_null          the gold has a value, the answer says null
  wrong_value        both have a value and they differ (dates counted above are excluded)
The point: these are the failures DPO pairs are built from (docs/DPO_PLAN.md WP0) and the ones the write-up should discuss.
"""
import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import jsonschema

SPLITS = ["test_a", "test_b", "test_c", "val", "test"]
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ECHO_KEYS = {"type", "required", "properties", "$schema", "additionalProperties", "enum", "format", "items", "$defs"}


def load_eval():
    spec = importlib.util.spec_from_file_location("ev", "eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def last_key(path):
    return re.split(r"[.\[]", path)[-1].rstrip("]") if path else path


def classify(ev, raw, gold, schema):
    """-> (document flags: Counter, leaf counts: Counter)"""
    flags, leaves = Counter(), Counter()
    text = (raw or "").strip()
    if text.startswith("```"):
        flags["fenced"] += 1
    ok, pred = ev.parse_json_object(raw)
    if not ok:
        flags["unparseable_unfinished" if text.lstrip("`json \n").startswith("{") and not text.rstrip().rstrip("`").rstrip().endswith("}") else "unparseable_other"] += 1
        return flags, leaves
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(pred))
    if errors:
        kind = errors[0].validator
        flags["schema_invalid"] += 1
        flags["schema_invalid:" + (kind if kind in ("required", "additionalProperties", "enum", "type") else "other")] += 1
    g, p = ev.flatten_json_to_leaves(gold), ev.flatten_json_to_leaves(pred)
    gold_keys = {last_key(k) for k in g}
    if any(last_key(k) in ECHO_KEYS and last_key(k) not in gold_keys for k in p if k not in g):
        flags["schema_echo"] += 1
    for path, gv in g.items():
        if path not in p:
            if gv is not None:
                leaves["missing_key"] += 1
            continue
        pv = p[path]
        if ev.leaves_match(gv, pv):
            continue
        if gv is None:
            leaves["invented_value"] += 1
        elif pv is None:
            leaves["over_null"] += 1
        elif isinstance(gv, str) and ISO.match(gv) and isinstance(pv, str) and not ISO.match(pv) and gv[:4] in pv:
            leaves["date_unnormalised"] += 1
        else:
            leaves["wrong_value"] += 1
    leaves["extra_key"] += sum(1 for k, v in p.items() if k not in g and v is not None)
    return flags, leaves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--dataset", default="sujalgawas/aya-json-sft-dataset")
    ap.add_argument("--systems", nargs="+")
    ap.add_argument("--splits", nargs="+")
    ap.add_argument("--json", help="also write the counts to this file")
    args = ap.parse_args()
    ev = load_eval()
    out, data = {}, {}
    for f in sorted(Path(args.results_dir).glob("*_*.json")):
        if f.name == "lexsi_provenance.json":
            continue
        split = next((s for s in SPLITS if f.stem.endswith("_" + s)), None)
        if split is None:
            continue
        system = f.stem[: -len(split) - 1]
        if (args.systems and system not in args.systems) or (args.splits and split not in args.splits):
            continue
        if split not in data:
            data[split] = {r["id"]: r for r in ev.load_split(split, args.data_dir, None, args.dataset)}
        rows, preds = data[split], json.load(open(f))["predictions"]
        flags, leaves = Counter(), Counter()
        for p in preds:
            r = rows[p["sample_id"]]
            a, b = classify(ev, p["raw_output"], r["gold"], r["schema"])
            flags.update(a)
            leaves.update(b)
        out[f"{system}/{split}"] = {"documents": len(preds), "flags": dict(flags), "leaves": dict(leaves)}
    if not out:
        raise SystemExit("nothing to analyse")
    keys = ["unparseable_unfinished", "unparseable_other", "fenced", "schema_invalid", "schema_invalid:required", "schema_invalid:additionalProperties",
            "schema_invalid:enum", "schema_invalid:type", "schema_echo"]
    lkeys = ["missing_key", "extra_key", "date_unnormalised", "invented_value", "over_null", "wrong_value"]
    print("documents with each failure (count and % of documents):")
    print(f"{'system/split':22s}{'n':>5s} " + " ".join(f"{k.replace('schema_invalid', 'sv').replace('unparseable_', 'unp_')[:11]:>12s}" for k in keys))
    for name, v in out.items():
        print(f"{name:22s}{v['documents']:5d} " + " ".join(f"{v['flags'].get(k, 0):5d} {100 * v['flags'].get(k, 0) / v['documents']:4.0f}%" for k in keys))
    print("\nleaf-level mistakes per 100 documents (on answers that parse):")
    print(f"{'system/split':22s}{'n':>5s} " + " ".join(f"{k:>17s}" for k in lkeys))
    for name, v in out.items():
        print(f"{name:22s}{v['documents']:5d} " + " ".join(f"{100 * v['leaves'].get(k, 0) / v['documents']:17.1f}" for k in lkeys))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print("\nwritten:", args.json)


if __name__ == "__main__":
    main()
