"""Leakage check for a train/val/test JSONL dataset (id, domain, schema, document, gold, schema_fp).

    python data/leakage_check.py data/aya-json-sft-dataset

Per AGENTS.md: a schema, or a near-duplicate of one, must never appear in more than one split.
Checks, train vs val and train vs test (and val vs test):
  1. same row id
  2. same schema_fp (the dataset's own schema hash)
  3. same schema after canonical JSON (sorted keys, "title" dropped)
  4. same set of field names (catches a schema renamed only in its title or description)
  5. near-duplicate field-name sets (Jaccard >= 0.8): a re-skinned schema
  6. same or near-same document text
  7. test gold string values that also appear in train gold (memorisation risk, reported not failed)
"""
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path


def load(d, name):
    rows = [json.loads(line) for line in open(Path(d) / f"{name}.jsonl", encoding="utf-8")]
    for r in rows:  # schema/gold may be stored as JSON strings or as objects
        for k in ("schema", "gold"):
            if isinstance(r[k], str):
                r[k] = json.loads(r[k])
    return rows


def canon(schema):
    s = {k: v for k, v in schema.items() if k not in ("title", "$schema")}
    return json.dumps(s, sort_keys=True)


def field_names(schema):
    out = set()

    def walk(n):
        if isinstance(n, dict):
            for k, v in n.get("properties", {}).items():
                out.add(k)
                walk(v)
            walk(n.get("items"))
        elif isinstance(n, list):
            for x in n:
                walk(x)

    walk(schema)
    return frozenset(out)


def doc_tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    else:
        yield o


def main(d):
    sp = {n: load(d, n) for n in ("train", "val", "test")}
    print({n: len(r) for n, r in sp.items()})
    for n, r in sp.items():
        print(f"  {n}: domains {dict(Counter(x['domain'] for x in r))}, sources {dict(Counter(x['source'] for x in r))}, "
              f"unique schemas (fp) {len({x['schema_fp'] for x in r})}, unique field-name sets {len({field_names(x['schema']) for x in r})}")

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        A, B = sp[a], sp[b]
        print(f"\n=== {a} vs {b} ===")
        print("same id:           ", len({x["id"] for x in A} & {x["id"] for x in B}))
        fa, fb = {x["schema_fp"] for x in A}, {x["schema_fp"] for x in B}
        print("same schema_fp:    ", len(fa & fb), "schemas;", sum(x["schema_fp"] in fa for x in B), f"of {len(B)} {b} rows")
        ca, cb = {canon(x["schema"]) for x in A}, {canon(x["schema"]) for x in B}
        print("same canon schema: ", len(ca & cb), "schemas;", sum(canon(x["schema"]) in ca for x in B), f"of {len(B)} {b} rows")
        na, nb = {field_names(x["schema"]) for x in A}, {field_names(x["schema"]) for x in B}
        print("same field names:  ", len(na & nb), "sets;", sum(field_names(x["schema"]) in na for x in B), f"of {len(B)} {b} rows")
        near = 0
        for x in B:
            fx = field_names(x["schema"])
            if any(len(fx & f) / max(len(fx | f), 1) >= 0.8 for f in na):
                near += 1
        print("near-dup (J>=.8):  ", near, f"of {len(B)} {b} rows")
        da = {x["document"] for x in A}
        print("same document:     ", sum(x["document"] in da for x in B), f"of {len(B)} {b} rows")
        toks = [doc_tokens(x["document"]) for x in A]
        nd = 0
        for x in B:
            t = doc_tokens(x["document"])
            if any(len(t & o) / max(len(t | o), 1) >= 0.9 for o in toks):
                nd += 1
        print("near-dup doc(J>=.9)", nd, f"of {len(B)} {b} rows")

    # value overlap: how many test gold string values were seen as a train gold value
    seen = {v for x in sp["train"] for v in leaves(x["gold"]) if isinstance(v, str)}
    for n in ("val", "test"):
        vals = [v for x in sp[n] for v in leaves(x["gold"]) if isinstance(v, str)]
        hit = sum(v in seen for v in vals)
        print(f"\n{n}: {hit}/{len(vals)} = {hit / max(len(vals), 1):.1%} gold string values also occur in train gold")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/aya-json-sft-dataset")
