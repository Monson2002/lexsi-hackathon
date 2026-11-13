"""Augment the aya-json-sft-dataset train rows so the model must NORMALISE and abstain, not just copy.

    python data/augment_train.py --in data/aya-json-sft-dataset/train.jsonl --out data/sft_v2/augmented.jsonl

Why (measured on train.jsonl, 1554 rows): every gold date, amount and enum value appears verbatim in its document,
documents only contain ISO dates, and required fields are never null. The SFT adapter learned "copy the text" and wrote
'July 26, 2017' where the schema asks for YYYY-MM-DD. Each op rewrites the DOCUMENT and keeps the gold exact:
  nulls   - delete the one sentence that holds a field's value; make that field nullable; gold becomes null
  dates   - ISO date in the document -> 'July 26, 2017' / '26/07/2025' ...; gold stays ISO; schema says 'Format: YYYY-MM-DD.'
  numbers - 148683.16 -> 148,683.16 (thousands separators) / 250.0 -> 250; gold stays numeric
  enums   - 'ONCE_DAILY' -> 'twice a day' style wording; gold stays the enum label
Every output row is re-validated: gold validates against its (edited) schema, and every non-null gold value is still
supported by the document (verbatim, or via the exact rendering this script produced). Rows that fail are dropped.
Gold stays exact by construction (AGENTS.md); no model is involved. Val/test files are never read or changed.
"""
import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

import jsonschema

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# Wording a human would use instead of the enum label. Falls back to de-underscoring + lower-casing.
ENUM_WORDS = {
    "ONCE_DAILY": ["once a day", "once daily", "one time per day"], "TWICE_DAILY": ["twice a day", "twice daily", "two times per day"],
    "STARTER": ["the Starter plan", "starter tier"], "GROWTH": ["the Growth plan", "growth tier"], "ENTERPRISE": ["the Enterprise plan", "enterprise tier"],
    "COMPLIANT": ["fully compliant", "in compliance"], "NON_COMPLIANT": ["not compliant", "non-compliant"], "CONDITIONAL": ["conditionally approved", "conditional"],
    "SEV1": ["Severity 1", "sev 1"], "SEV2": ["Severity 2", "sev 2"], "SEV3": ["Severity 3", "sev 3"], "SEV4": ["Severity 4", "sev 4"],
    "PAID": ["paid in full", "settled"], "PENDING": ["awaiting payment", "pending"], "OVERDUE": ["past due", "overdue"],
    "us-west-2": ["US West (Oregon)"], "us-east-1": ["US East (N. Virginia)"], "eu-west-1": ["EU West (Ireland)"], "ap-south-1": ["Asia Pacific (Mumbai)"],
    "USD": ["US dollars"], "EUR": ["euros"], "GBP": ["pounds sterling"], "CAD": ["Canadian dollars"], "INR": ["Indian rupees"],
}


def leaves(schema, gold, path=()):
    """(path, subschema, value) for every leaf; path = keys and list indices."""
    if isinstance(gold, dict):
        for k, v in gold.items():
            yield from leaves((schema or {}).get("properties", {}).get(k, {}), v, path + (k,))
    elif isinstance(gold, list):
        for i, v in enumerate(gold):
            yield from leaves((schema or {}).get("items", {}), v, path + (i,))
    else:
        yield path, schema, gold


def sub_at(schema, path):
    for p in path:
        schema = schema["items"] if isinstance(p, int) else schema["properties"][p]
    return schema


def set_at(gold, path, value):
    for p in path[:-1]:
        gold = gold[p]
    gold[path[-1]] = value


def make_nullable(sub):
    if isinstance(sub.get("type"), str):
        sub["type"] = [sub["type"], "null"]
    elif isinstance(sub.get("type"), list) and "null" not in sub["type"]:
        sub["type"] = sub["type"] + ["null"]
    if "enum" in sub and None not in sub["enum"]:
        sub["enum"] = sub["enum"] + [None]


def rep(v):
    return repr(v) if isinstance(v, float) else str(v)


def fmt_date(iso, rng):
    y, m, d = map(int, iso.split("-"))
    date(y, m, d)  # raises on an impossible date
    opts = [f"{MONTHS[m - 1]} {d}, {y}", f"{d} {MONTHS[m - 1]} {y}", f"{MONTHS[m - 1][:3]} {d}, {y}", f"{d:02d}-{MONTHS[m - 1][:3]}-{y}"]
    if d > 12:  # numeric day-first / month-first is only unambiguous when the day cannot be a month
        opts += [f"{d:02d}/{m:02d}/{y}", f"{m:02d}/{d:02d}/{y}", f"{d:02d}.{m:02d}.{y}"]
    return rng.choice(opts)


def split_segments(doc):
    """Sentences / lines, keeping their exact text so one can be cut out of the document."""
    return [s for s in re.split(r"(?<=[.!?])\s+|\n", doc) if s.strip()]


def op_nulls(rec, rng, rendered):
    """Remove one sentence per chosen field; that field becomes null (and nullable if it was not)."""
    doc, gold, schema = rec["document"], rec["gold"], rec["schema"]
    all_leaves = [(p, s, v) for p, s, v in leaves(schema, gold) if v is not None]
    segs = split_segments(doc)
    cand = []
    for path, sub, v in all_leaves:
        if isinstance(v, bool) or any(isinstance(x, int) for x in path):  # skip bools and array items
            continue
        hits = [s for s in segs[1:] if rep(v) in s]  # segs[0] is the headline (id, date)
        if len(hits) != 1 or doc.count(rep(v)) != 1:
            continue
        seg = hits[0]
        others = [rep(ov) for op_, _, ov in all_leaves if op_ != path and not isinstance(ov, bool)]
        # word-boundary match, so a short value such as 68 still blocks the cut but does not fire inside '1068'
        if any(re.search(r"(?<![\w.])" + re.escape(o) + r"(?![\w])", seg) for o in others) or re.search(r"\b(true|false|yes|no)\b", seg, re.I):
            continue  # the sentence carries another field (or a boolean we cannot see): do not cut it
        cand.append((path, seg))
    rng.shuffle(cand)
    done = 0
    for path, seg in cand[: rng.choice([1, 2, 3])]:
        if seg not in rec["document"]:
            continue
        cut = re.sub(r"[ \t]{2,}", " ", rec["document"].replace(seg, "", 1))
        rec["document"] = re.sub(r"[ \t]+\n", "\n", cut).replace("\n\n\n", "\n\n").strip()
        make_nullable(sub_at(rec["schema"], path))
        set_at(rec["gold"], path, None)
        done += 1
    return done


def op_dates(rec, rng, rendered):
    n = 0
    for path, sub, v in list(leaves(rec["schema"], rec["gold"])):
        if not (isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)) or v not in rec["document"]:
            continue
        try:
            new = fmt_date(v, rng)
        except ValueError:
            continue
        rec["document"] = rec["document"].replace(v, new)
        rendered[path] = new
        sub["format"] = "date"
        if "YYYY-MM-DD" not in sub.get("description", ""):
            sub["description"] = (sub.get("description", "") + " Format: YYYY-MM-DD.").strip()
        n += 1
    return n


def op_numbers(rec, rng, rendered):
    n = 0
    for path, sub, v in list(leaves(rec["schema"], rec["gold"])):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or path in rendered:
            continue
        tok = rep(v)
        pat = re.compile(r"(?<![\d.,\w-])" + re.escape(tok) + r"(?![\d\w])")
        if len(pat.findall(rec["document"])) != 1:
            continue
        if abs(v) >= 1000:
            new = f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
        elif isinstance(v, float) and v == int(v) and rng.random() < 0.7:
            new = str(int(v))
        else:
            continue
        rec["document"] = pat.sub(new, rec["document"], count=1)
        rendered[path] = new
        n += 1
    return n


def op_enums(rec, rng, rendered):
    n = 0
    for path, sub, v in list(leaves(rec["schema"], rec["gold"])):
        if not isinstance(v, str) or "enum" not in sub or path in rendered:
            continue
        pat = re.compile(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])")
        if len(pat.findall(rec["document"])) != 1:
            continue
        new = rng.choice(ENUM_WORDS.get(v, [v.replace("_", " ").lower()]))
        if new == v:
            continue
        rec["document"] = pat.sub(new, rec["document"], count=1)
        rendered[path] = new
        n += 1
    return n


def supported(rec, rendered):
    """Every non-null string/number gold value must appear in the document verbatim, or as the rendering we produced."""
    doc = rec["document"]
    for path, _, v in leaves(rec["schema"], rec["gold"]):
        if v is None or isinstance(v, bool):
            continue
        if path in rendered:
            if rendered[path] not in doc:
                return False
        elif rep(v) not in doc:
            return False
    return True


def fp(schema):
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/aya-json-sft-dataset/train.jsonl")
    ap.add_argument("--out", default="data/sft_v2/augmented.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copies", type=int, default=1, help="augmented rows per source row")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8")]
    rng = random.Random(args.seed)
    OPS = [("nulls", op_nulls, 0.6), ("dates", op_dates, 0.9), ("numbers", op_numbers, 0.6), ("enums", op_enums, 0.7)]
    out, stats = [], Counter()
    for r in rows:
        for c in range(args.copies):
            rec = {"schema": copy.deepcopy(r["schema"]), "document": r["document"], "gold": copy.deepcopy(r["gold"])}
            rendered, applied = {}, []
            for name, fn, p in OPS:
                if rng.random() < p and fn(rec, rng, rendered):
                    applied.append(name)
            if not applied:
                stats["no_op_applicable"] += 1
                continue
            try:
                jsonschema.Draft202012Validator(rec["schema"]).validate(rec["gold"])
            except jsonschema.ValidationError:
                stats["dropped_gold_invalid"] += 1
                continue
            if not supported(rec, rendered):
                stats["dropped_unsupported_value"] += 1
                continue
            for a in applied:
                stats[f"op:{a}"] += 1
            out.append({"id": f"{r['id']}_aug{c}", "domain": r["domain"], "source": "augmented:" + "+".join(applied), "schema": rec["schema"],
                        "document": rec["document"], "gold": rec["gold"], "schema_fp": fp(rec["schema"]),
                        "provenance": {"base_id": r["id"], "ops": applied, "seed": args.seed}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"{len(rows)} source rows -> {len(out)} augmented rows -> {args.out}")
    print(dict(stats))


if __name__ == "__main__":
    main()
