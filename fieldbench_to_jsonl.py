"""Convert the FieldBench corpus into our JSONL format (id, domain, schema, document, gold, split, provenance).

    git clone --depth 1 https://github.com/fieldbench/corpus
    pip install pyyaml jsonschema
    python fieldbench_to_jsonl.py --corpus corpus --out fieldbench_out
    python eval.py --data-dir fieldbench_out --splits test_b test_c --systems base sft grpo

Why two splits (AGENTS.md: split by domain, never leak):
- test_b = held-out DOMAIN. Our train data covers invoices / support tickets / clinical notes / forms. FieldBench's
  insurance, contracts, legal and SEC categories are domains the model never saw. Synthetic documents only (CC0),
  because the real ones in those categories either have no explicit licence or are too long for an 8K model.
- test_c = REAL public documents with a clear licence (CC BY / public domain / CC0), any category.
Real vs synthetic stays visible in every record (`source`) because FieldBench's datasheet warns that synthetic
documents overestimate accuracy, so never quote one pooled number.

Deliberately NOT used: real medical_records (MTSamples: no redistribution licence, and our rule is no real patient
data), real EDGAR filings (no explicit licence), documents too long for the model. Counts of each are printed.
FieldBench is an evaluation set: do not train on it (that would remove the held-out test).
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import jsonschema
import yaml

# ponytail: knobs, edit here. Our train domains: invoices, support_tickets, clinical_notes, forms_reports.
TEST_B_CATEGORIES = {"insurance_certificates", "insurance_policies", "insurance_claims", "contracts", "legal_filings", "sec_filings"}
TEST_C_CATEGORIES = {"receipts", "legal_filings", "insurance_claims"}
OK_LICENSES = ("CC BY 4.0", "CC0", "Public domain", "Public Domain")  # prefix match on the manifest `license`
MAX_DOC_CHARS = 4000  # ~1,000 tokens: same cap as Sujal's training data (MAX_DOC_TOKENS=1000), and leaves room for the schema


def field_schema(f):
    """FieldBench YAML field -> JSON Schema property. Every property is nullable (a missing value is `null`)."""
    t = f["type"]
    desc = " ".join((f.get("description") or "").split())[:300]  # first 300 chars; the long hints/patterns are dropped
    if t == "date":
        s = {"type": ["string", "null"], "format": "date"}
        desc = (desc + " Format: YYYY-MM-DD.").strip()
    elif t == "enum":
        s = {"type": ["string", "null"], "enum": list(f["options"]) + [None]}
    elif t == "number":
        s = {"type": ["number", "null"]}
    elif t == "array":
        item = f.get("items") or {"type": "string"}
        if item.get("type") == "object":
            props = {k: field_schema(v) for k, v in item["properties"].items()}
            s = {"type": ["array", "null"], "items": {
                "type": "object", "additionalProperties": False, "required": list(props), "properties": props}}
        else:
            s = {"type": ["array", "null"], "items": field_schema(item)}
    else:  # string and anything unknown
        s = {"type": ["string", "null"]}
    if desc:
        s["description"] = desc
    return s


def to_json_schema(yaml_schema):
    props = {k: field_schema(v) for k, v in yaml_schema["fields"].items()}
    # All keys required + all values nullable: the model must decide every field, and `null` is the explicit "absent".
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": yaml_schema["name"], "type": "object",
            "additionalProperties": False, "required": list(props), "properties": props}


def fill_gold(expected, schema):
    """Absent keys in FieldBench's expected JSON mean 'not in the document'; our gold spells that out as null."""
    gold = {k: expected.get(k) for k in schema["properties"]}
    for k, p in schema["properties"].items():
        if isinstance(gold[k], list) and isinstance(p.get("items"), dict) and "properties" in p["items"]:
            gold[k] = [{ik: it.get(ik) for ik in p["items"]["properties"]} if isinstance(it, dict) else it for it in gold[k]]
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus", help="path to a clone of github.com/fieldbench/corpus")
    ap.add_argument("--out", default="fieldbench_out")
    args = ap.parse_args()
    root, out = Path(args.corpus), Path(args.out)
    out.mkdir(exist_ok=True)
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = ""

    rows = {"test_b": [], "test_c": []}
    skipped = Counter()
    for cat_dir in sorted(p for p in root.iterdir() if (p / "documents").is_dir()):
        cat = cat_dir.name
        for doc_path in sorted((cat_dir / "documents").glob("*.md")):
            did = doc_path.stem
            man = json.load(open(cat_dir / "manifests" / f"{did}.json"))
            source = man["source"]
            license_ok = man["license"].startswith(OK_LICENSES)
            split = ("test_b" if source == "synthetic" and cat in TEST_B_CATEGORIES else
                     "test_c" if source == "real" and cat in TEST_C_CATEGORIES else None)
            if split is None:
                skipped[f"{cat}/{source}: not in test_b or test_c"] += 1
                continue
            if not license_ok:
                skipped[f"{cat}/{source}: licence not allowed"] += 1
                continue
            doc = doc_path.read_text(encoding="utf-8")
            if len(doc) > MAX_DOC_CHARS:
                skipped[f"{cat}/{source}: too long"] += 1
                continue
            yaml_schema = yaml.safe_load(open(cat_dir / "schemas" / Path(man["schema"]).name))
            schema = to_json_schema(yaml_schema)
            gold = fill_gold(json.load(open(cat_dir / "expected" / f"{did}.expected.json")), schema)
            try:  # gold must satisfy the schema we hand the model, or schema_valid could never reach 1.0
                jsonschema.validate(gold, schema)
            except jsonschema.ValidationError as e:
                skipped[f"{cat}/{source}: gold fails converted schema ({e.message[:60]})"] += 1
                continue
            rows[split].append({
                "id": f"fb_{cat}_{did}", "domain": cat, "source": source, "split": split,
                "schema_id": f"fb_{yaml_schema['name']}", "schema": schema, "document": doc, "gold": gold,
                "provenance": {"dataset": "fieldbench/corpus", "commit": sha, "license": man["license"],
                               "attribution": man.get("attribution"), "source_name": man.get("source_name"),
                               "doc_id": f"{cat}/{did}"},
            })

    for split, rs in rows.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        nulls = sum(v is None for r in rs for v in r["gold"].values())
        slots = sum(len(r["gold"]) for r in rs)
        print(f"{split}: {len(rs)} docs {dict(Counter((r['domain'], r['source']) for r in rs))} | top-level null slots {nulls}/{slots}")
    print("\nskipped:")
    for k, v in sorted(skipped.items()):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
