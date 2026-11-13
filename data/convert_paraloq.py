"""Convert paraloq/json_data_extraction (484 rows: text + JSON schema + item) to our training format.

    python data/convert_paraloq.py --out data/sft_v2/paraloq.jsonl

Licence: the Hub tag says Apache-2.0, but the card's 'Out-of-Scope Use' says 'Intended for research purposes only'.
The two conflict, so every row carries source='paraloq/json_data_extraction' and can be dropped with one filter if the team
decides not to ship it. The dataset is item-first (the text was written from the item), which matches AGENTS.md's schema-first rule.

What this script changes (and why):
  - our convention: every property required and nullable, additionalProperties false, missing item keys become null
    (the dataset itself has almost no nulls: 0.1% of leaves)
  - keeps $defs/$ref untouched in structure (research: $defs/$ref is what makes small models echo the schema)
  - values the text does not support become null (<=2 per row and <=20% of its values); rows with more are dropped
    (dates may be written in another format: year, day and month must all be present)
"""
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import jsonschema

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


def nullable_node(n):
    if not isinstance(n, dict):
        return n
    if "$ref" in n:
        return {"anyOf": [n, {"type": "null"}]}
    for key in ("anyOf", "oneOf"):
        if key in n:
            if not any(isinstance(b, dict) and b.get("type") == "null" for b in n[key]):
                n[key] = n[key] + [{"type": "null"}]
            return n
    t = n.get("type")
    if isinstance(t, str):
        n["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        n["type"] = t + ["null"]
    if "enum" in n and None not in n["enum"]:
        n["enum"] = n["enum"] + [None]
    return n


def to_ours(node):
    """In place: objects get every property required + nullable and additionalProperties false."""
    if not isinstance(node, dict):
        return
    if "properties" in node:
        node["properties"] = {k: nullable_node(v) for k, v in node["properties"].items()}
        node["required"] = list(node["properties"])
        node["additionalProperties"] = False
        for v in node["properties"].values():
            to_ours(v)
    for key in ("items", "$defs", "definitions"):
        v = node.get(key)
        if key == "items":
            to_ours(v)
        elif isinstance(v, dict):
            for d in v.values():
                to_ours(d)
    for key in ("anyOf", "oneOf"):
        for b in node.get(key, []):
            to_ours(b)


def deref(sub, root):
    while isinstance(sub, dict):
        if "$ref" in sub:
            cur = root
            for part in sub["$ref"][2:].split("/"):
                cur = cur[part]
            sub = cur
        elif "anyOf" in sub or "oneOf" in sub:
            branches = [b for b in sub.get("anyOf", sub.get("oneOf")) if b.get("type") != "null"]
            sub = branches[0] if branches else {}
        else:
            break
    return sub


def fill(g, sub, root):
    """Item keys the dataset omitted become explicit nulls (our convention)."""
    sub = deref(sub, root)
    if isinstance(g, dict) and isinstance(sub, dict) and "properties" in sub:
        out = {k: fill(g[k], v, root) if k in g else None for k, v in sub["properties"].items()}
        out.update({k: v for k, v in g.items() if k not in out})  # extra keys stay; validation will then reject the row
        return out
    if isinstance(g, list) and isinstance(sub, dict) and "items" in sub:
        return [fill(x, sub["items"], root) for x in g]
    return g


def norm_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(s).lower())).strip()


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    else:
        yield o


def path_leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from path_leaves(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from path_leaves(v, path + (i,))
    else:
        yield path, o


def unsupported(item, text):
    """Paths of non-null string/number values that the text does not support."""
    nt = norm_text(text)
    low = text.lower()
    bad = []
    for path, v in path_leaves(item):
        if v is None or isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            toks = {str(v), f"{v:,}", f"{v:,.2f}", str(int(v)) if float(v) == int(v) else str(v)}
            ok = any(t in text for t in toks)
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            y, m, d = v.split("-")
            ok = v in text or (y in text and re.search(rf"\b0?{int(d)}\b", text) and (MONTHS[int(m) - 1] in low or MONTHS[int(m) - 1][:3] in low or re.search(rf"\b0?{int(m)}\b", text)))
        else:
            ok = norm_text(v) in nt
        if not ok:
            bad.append(path)
    return bad


def truly_absent(v, text):
    """No word / number of the value appears in the text at all. A paraphrase or a different date format is NOT absent:
    nulling it would teach the model to say null for information that is in the document."""
    toks = re.findall(r"[A-Za-z0-9]+", str(v))
    low = text.lower()
    return bool(toks) and not any(t.lower() in low for t in toks if len(t) >= 2) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", str(v))


def fp(schema):
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v2/paraloq.jsonl")
    args = ap.parse_args()
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("paraloq/json_data_extraction", "data.parquet", repo_type="dataset"))
    out, stats = [], Counter()
    for i, r in df.iterrows():
        try:
            schema, item = json.loads(r["schema"]), json.loads(r["item"])
        except Exception:
            stats["dropped_unparseable"] += 1
            continue
        if len(r["text"]) > 4200:
            stats["dropped_too_long"] += 1
            continue
        schema = copy.deepcopy(schema)
        to_ours(schema)
        schema.setdefault("title", r["title"])
        gold = fill(item, schema, schema)
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.Draft202012Validator(schema).validate(gold)
        except Exception:
            stats["dropped_invalid_after_conversion"] += 1
            continue
        bad = unsupported(gold, r["text"])
        total = sum(1 for _, v in path_leaves(gold) if v is not None and not isinstance(v, bool))
        if bad:
            # not in the document -> null (AGENTS.md). Only for a few values; arrays of scalars cannot hold null.
            vals = []
            for p in bad:
                tgt = gold
                for k in p:
                    tgt = tgt[k]
                vals.append(tgt)
            if len(bad) > 2 or len(bad) > 0.2 * total or any(isinstance(p[-1], int) for p in bad) or not all(truly_absent(v, r["text"]) for v in vals):
                stats["dropped_value_not_in_text"] += 1
                continue
            for p in bad:
                tgt = gold
                for k in p[:-1]:
                    tgt = tgt[k]
                tgt[p[-1]] = None
            try:
                jsonschema.Draft202012Validator(schema).validate(gold)
            except Exception:
                stats["dropped_invalid_after_nulling"] += 1
                continue
            stats["kept_with_nulled_values"] += 1
        stats["kept"] += 1
        out.append({"id": f"paraloq_{r['item_id']}_{i}", "domain": f"paraloq/{r['topic']}", "source": "paraloq/json_data_extraction",
                    "schema": schema, "document": r["text"], "gold": gold, "schema_fp": fp(schema),
                    "provenance": {"dataset": "paraloq/json_data_extraction", "medium": r["medium"], "nulled_paths": ["/".join(map(str, p)) for p in bad],
                                   "licence_note": "Hub tag Apache-2.0; card also says 'research purposes only' (unresolved)"}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"{len(df)} rows -> {len(out)} kept -> {args.out}")
    print(dict(stats))


if __name__ == "__main__":
    main()
