"""Schema-first training data from real-world JSON Schemas (epfl-dlab/JSONSchemaBench, MIT dataset card).

    python data/gen_schema_first.py --out data/sft_v2/schema_first.jsonl

Pipeline per schema (AGENTS.md: gold is built schema-first, so it is exact by construction; no LLM is used):
  1. normalise the real schema to the subset we support (resolve local $ref, collapse anyOf/oneOf/allOf, drop pattern etc.)
     and apply OUR convention: every property required, every value nullable ('null' = absent), additionalProperties false
  2. sample a gold record that satisfies it; 15-25% of leaves become null (absent from the document, or shown as N/A)
  3. render a document in one of several styles; dates in random formats (gold stays YYYY-MM-DD), amounts with
     separators, enum labels worded like a person would write them, distractor values, filler text
  4. in half the schemas lift nested objects into $defs/$ref (research: $defs/$ref triggers schema echo in small models)
  5. verify: gold validates against the final schema, and every non-null value is supported by the document
Only the schemas' STRUCTURE (field names, nesting, enums, types) comes from JSONSchemaBench; the schemas themselves keep
their upstream licences (listed in the dataset card of JSONSchemaBench). FieldBench is not used.
"""
import argparse
import copy
import hashlib
import json
import random
import re
import uuid
from collections import Counter
from pathlib import Path

import jsonschema
from faker import Faker

SUBSETS = ["Github_easy", "Github_medium", "Snowplow", "WashingtonPost", "Glaiveai2K", "Github_trivial"]
FORMATS = {"date", "date-time", "email", "uri", "uuid"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
FILLER = ["This record was generated automatically; please retain it for your files.", "Contact the issuing office if any detail looks wrong.",
          "Page 1 of 1. Internal use only.", "Reference copy, no signature required.", "Scanned and indexed by the records team."]


# ------------------------------------------------------------------ 1. normalise a real-world schema
def resolve(node, root, seen=()):
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            return None
        cur = root
        for part in ref[2:].split("/"):
            cur = cur.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(cur, dict) else None
            if cur is None:
                return None
        seen, node = seen + (ref,), cur
    return node


def norm(node, root, depth=0, seen=()):
    """Real schema node -> supported node (not yet nullable), or None if it cannot be supported."""
    node = resolve(node, root, seen)
    if not isinstance(node, dict) or depth > 4:
        return None
    node = dict(node)
    for key in ("allOf", "anyOf", "oneOf"):
        if key in node and isinstance(node[key], list):
            branches = [resolve(b, root, seen) for b in node.pop(key)]
            branches = [b for b in branches if isinstance(b, dict) and b.get("type") != "null"]
            if not branches:
                return None
            for b in (branches if key == "allOf" else branches[:1]):
                for k, v in b.items():
                    if k == "properties":
                        node.setdefault("properties", {}).update(v)
                    else:
                        node.setdefault(k, v)
    if "const" in node:
        node["enum"] = [node.pop("const")]
    t = node.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t is None:
        t = "object" if "properties" in node else "array" if "items" in node else (
            "integer" if node.get("enum") and all(isinstance(v, int) and not isinstance(v, bool) for v in node["enum"]) else "string")
    out = {"type": t}
    if node.get("description"):
        out["description"] = " ".join(str(node["description"]).split())[:200]
    if t == "object":
        props = {}
        for k, v in list((node.get("properties") or {}).items())[:30]:
            n = norm(v, root, depth + 1, seen)
            if n is not None:
                props[k] = n
        if not props:
            return None
        out.update(properties=props, required=list(props), additionalProperties=False)
    elif t == "array":
        item = norm(node.get("items") or {"type": "string"}, root, depth + 1, seen) or {"type": "string"}
        if item["type"] == "array":
            item = {"type": "string"}
        if item["type"] == "object":  # the renderer prints one line per list item, so items keep scalar fields only
            item["properties"] = {k: v for k, v in item["properties"].items() if v["type"] not in ("object", "array")}
            if not item["properties"]:
                item = {"type": "string"}
            else:
                item["required"] = list(item["properties"])
        out["items"] = item
        out["minItems"], out["maxItems"] = 1, min(int(node.get("maxItems") or 3), 3)
    elif t in ("string", "integer", "number", "boolean"):
        en = [v for v in node.get("enum", []) if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        if en:
            out["enum"] = en[:12]
        if node.get("format") in FORMATS:
            out["format"] = node["format"]
        for k in ("minimum", "maximum", "minLength", "maxLength"):
            if isinstance(node.get(k), (int, float)):
                out[k] = node[k]
    else:
        return None
    return out


def nullable(node):
    """Our convention: every value may be null ('absent')."""
    node = copy.deepcopy(node)

    def go(n):
        n["type"] = [n["type"], "null"]
        if "enum" in n:
            n["enum"] = n["enum"] + [None]
        for v in n.get("properties", {}).values():
            go(v)
        if "items" in n:
            go_items(n["items"])

    def go_items(n):  # array elements stay non-null; only their children are nullable
        for v in n.get("properties", {}).values():
            go(v)
        if "items" in n:
            go_items(n["items"])

    for v in node.get("properties", {}).values():
        go(v)
    return node


def size(node):
    return 1 + sum(size(v) for v in node.get("properties", {}).values()) + (size(node["items"]) if "items" in node else 0)


def depth_of(node):
    return 1 + max([depth_of(v) for v in node.get("properties", {}).values()] + [depth_of(node["items"]) if "items" in node else 0])


def camel(key):
    return "".join(w.capitalize() for w in re.split(r"[^A-Za-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", key)) if w) or "Item"


def lift_defs(schema):
    """Move nested objects into $defs and reference them (nullable ones as anyOf[$ref, null])."""
    defs = {}

    def unique(name):
        base, i = name, 2
        while name in defs:
            name, i = f"{base}{i}", i + 1
        return name

    def visit(props):
        for k, sub in props.items():
            tt = sub["type"]
            tl = tt if isinstance(tt, list) else [tt]
            if "object" in tl:
                visit(sub["properties"])
                name = unique(camel(k))
                body = {x: y for x, y in sub.items() if x != "type"}
                defs[name] = {"type": "object", **body}
                props[k] = {"anyOf": [{"$ref": f"#/$defs/{name}"}, {"type": "null"}]}
            elif "array" in tl and sub["items"].get("type") == "object":
                visit(sub["items"]["properties"])
                name = unique(camel(k) + "Item")
                defs[name] = dict(sub["items"])
                sub["items"] = {"$ref": f"#/$defs/{name}"}

    visit(schema["properties"])
    if defs:
        schema["$defs"] = defs
    return schema


# ------------------------------------------------------------------ 2. sample gold + the text each value will have
def humanize(key):
    return re.sub(r"[\s_.\-]+", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)).strip().lower()


def fmt_date(iso, rng):
    y, m, d = map(int, iso.split("-"))
    opts = [f"{MONTHS[m - 1]} {d}, {y}", f"{d} {MONTHS[m - 1]} {y}", f"{MONTHS[m - 1][:3]} {d}, {y}", f"{d:02d}-{MONTHS[m - 1][:3]}-{y}"]
    if d > 12:
        opts += [f"{d:02d}/{m:02d}/{y}", f"{m:02d}/{d:02d}/{y}", f"{d:02d}.{m:02d}.{y}"]
    return rng.choice(opts)


def money(v, rng):
    return f"{v:,.2f}" if isinstance(v, float) and abs(v) >= 1000 and rng.random() < 0.8 else (f"{v:,}" if isinstance(v, int) and abs(v) >= 1000 and rng.random() < 0.8 else None)


def enum_text(v, rng):
    if not isinstance(v, str):
        return str(v)
    if re.fullmatch(r"[A-Z0-9_]+", v) and rng.random() < 0.8:
        return rng.choice([v.replace("_", " ").lower(), v.replace("_", " ").title()])
    return v


def string_value(key, sub, fk, rng):
    k, fmt = key.lower(), sub.get("format")
    if fmt == "date":
        return fk.date_between("-6y", "+1y").isoformat()
    if fmt == "date-time":
        return fk.date_time_between("-3y", "now").strftime("%Y-%m-%dT%H:%M:%SZ")
    if fmt == "email" or "email" in k:
        return fk.email()
    if fmt == "uri" or re.search(r"url|uri|link|website|href", k):
        return fk.url()
    if fmt == "uuid":
        return str(uuid.UUID(int=rng.getrandbits(128)))
    if "date" in k or k.endswith("_at") or k.endswith("at"):
        return fk.date_between("-6y", "+1y").isoformat()
    if re.search(r"phone|tel|fax", k):
        return fk.phone_number()
    if re.search(r"first.?name", k):
        return fk.first_name()
    if re.search(r"last.?name|surname", k):
        return fk.last_name()
    if re.search(r"company|organi[sz]ation|vendor|client|customer|supplier|employer|publisher", k):
        return fk.company()
    if "name" in k or re.search(r"author|owner|user|person|contact|manager", k):
        return fk.name()
    if "city" in k:
        return fk.city()
    if "country" in k:
        return fk.country()
    if "address" in k:
        return fk.street_address()
    if re.search(r"version", k):
        return f"{rng.randint(0, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 30)}"
    if re.search(r"(^|_|\b)(id|code|ref|key|token|sku|serial|number|no)\b|id$|code$|ref$", k):
        return f"{fk.lexify('???').upper()}-{rng.randint(1000, 99999)}"
    if re.search(r"path|file|dir|folder", k):
        return "/" + "/".join(fk.words(rng.randint(1, 3)))
    if re.search(r"title|summary|description|text|message|comment|note|content|reason|label|subject|body", k):
        return fk.sentence(nb_words=rng.randint(4, 9)).rstrip(".")
    return " ".join(fk.words(rng.randint(1, 3)))


def sample(sub, key, path, fk, rng, texts, absent_p):
    """Gold value for one node; fills texts[path] with the text the document will use for each scalar."""
    t = [x for x in (sub["type"] if isinstance(sub["type"], list) else [sub["type"]]) if x != "null"][0]
    if t == "object":
        return {k: sample(v, k, path + (k,), fk, rng, texts, absent_p) for k, v in sub["properties"].items()}
    if t == "array":
        item = sub["items"]
        return [sample(item, key, path + (i,), fk, rng, texts, absent_p) for i in range(rng.randint(sub.get("minItems", 1), sub.get("maxItems", 3)))]
    if t != "object" and path and rng.random() < absent_p and len(path) >= 1 and not isinstance(path[-1], int):
        if rng.random() < 0.3:
            texts[path] = ("NA", rng.choice(["N/A", "not provided", "unknown", "-"]))
        return None
    if "enum" in sub:
        v = rng.choice([e for e in sub["enum"] if e is not None])
        texts[path] = ("V", enum_text(v, rng))
        return v
    if t == "boolean":
        v = rng.random() < 0.5
        texts[path] = ("B", ("yes" if v else "no") if rng.random() < 0.5 else ("true" if v else "false"))
        return v
    if t in ("integer", "number"):
        lo, hi = sub.get("minimum", 0), sub.get("maximum", 5000)
        if hi < lo:
            hi = lo + 100
        v = rng.randint(int(lo), int(hi)) if t == "integer" else round(rng.uniform(lo, min(hi, lo + 20000)), 2)
        texts[path] = ("N", money(v, rng) or str(v))
        return v
    v = string_value(key, sub, fk, rng)
    if "maxLength" in sub and len(v) > sub["maxLength"]:
        v = v[: int(sub["maxLength"])].rstrip()
    if sub.get("format") == "date" or re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):  # incl. fields only named like dates
        texts[path] = ("D", fmt_date(v, rng))
    else:
        texts[path] = ("S", v)
    return v


# ------------------------------------------------------------------ 3. render a document
def label(key, rng, mode):
    h = humanize(key)
    return {"upper": h.upper(), "title": h.title(), "raw": key}.get(mode, h)


def render(gold, texts, schema, rng, style, mode, title):
    lines = []

    def leaf_text(path):
        return texts.get(path, (None, None))

    def walk(g, sub, path, indent):
        if isinstance(g, dict):
            for k in rng.sample(list(g), len(g)) if style != "table" else list(g):
                v = g[k]
                p = path + (k,)
                if isinstance(v, dict):
                    lines.append(f"{'  ' * indent}{label(k, rng, mode)}:")
                    walk(v, None, p, indent + 1)
                elif isinstance(v, list):
                    emit_list(k, v, p, indent)
                elif v is None:
                    kind, txt = leaf_text(p)
                    if kind == "NA":
                        lines.append(emit(k, txt, indent))
                else:
                    lines.append(emit(k, leaf_text(p)[1], indent))

    def emit(k, txt, indent):
        lab = label(k, rng, mode)
        if style == "prose":
            return f"{'  ' * indent}The {lab} is {txt}."
        if style == "table":
            return f"{'  ' * indent}| {lab} | {txt} |"
        return f"{'  ' * indent}{lab}: {txt}"

    def emit_list(k, items, p, indent):
        if all(not isinstance(x, (dict, list)) for x in items):
            lines.append(emit(k, ", ".join(texts[p + (i,)][1] for i in range(len(items))), indent))
            return
        lines.append(f"{'  ' * indent}{label(k, rng, mode)}:")
        for i, it in enumerate(items, 1):
            parts = []
            for kk, vv in it.items():
                if vv is None or isinstance(vv, (dict, list)):
                    continue
                parts.append(f"{label(kk, rng, mode)}: {texts[p + (i - 1, kk)][1]}")
            lines.append(f"{'  ' * (indent + 1)}{i}. " + "; ".join(parts))

    walk(gold, schema, (), 0)
    body = "\n".join(lines)
    head = f"{rng.choice(['Record', 'Document', 'Entry', 'Report', 'Notice'])}: {title}\n\n"
    tail = ("\n\n" + rng.choice(FILLER)) if rng.random() < 0.5 else ""
    return head + body + tail


def add_distractor(doc, gold, texts, rng):
    """A superseded date/amount with a clear 'do not use' label; never equal to a gold text."""
    cand = [(p, texts[p]) for p in texts if texts[p][0] in ("D", "N")]
    if not cand:
        return doc
    p, (kind, txt) = rng.choice(cand)
    other = fmt_date(f"{rng.randint(2018, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", rng) if kind == "D" else f"{rng.randint(10, 999)}.{rng.randint(10, 99)}"
    if any(other == t for _, t in texts.values()):
        return doc
    lab = humanize(str([x for x in p if isinstance(x, str)][-1])) if any(isinstance(x, str) for x in p) else "value"
    return doc + f"\nNote: an earlier draft listed the {lab} as {other}; that figure was superseded and must not be used."


# ------------------------------------------------------------------ 5. verify
def supported(gold, texts, doc, path=()):
    if isinstance(gold, dict):
        return all(supported(v, texts, doc, path + (k,)) for k, v in gold.items())
    if isinstance(gold, list):
        return all(supported(v, texts, doc, path + (i,)) for i, v in enumerate(gold))
    if gold is None or isinstance(gold, bool):
        return True
    return path in texts and texts[path][1] in doc


def fp(schema):
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def load_schemas(subsets, per_subset_cap, split="train"):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    seen, out = set(), []
    for s in subsets:
        df = pd.read_parquet(hf_hub_download("epfl-dlab/JSONSchemaBench", f"{s}/{split}-00000-of-00001.parquet", repo_type="dataset"))
        n = 0
        for uid, txt in zip(df.unique_id, df.json_schema):
            try:
                raw = json.loads(txt)
            except Exception:
                continue
            sc = norm(raw, raw)
            if not sc or sc["type"] != "object" or not (3 <= len(sc["properties"]) <= 25) or size(sc) > 45 or depth_of(sc) > 4:
                continue
            h = fp(sc)
            if h in seen:
                continue
            seen.add(h)
            sc["title"] = str(raw.get("title") or s)[:60]
            out.append((s, uid, sc))
            n += 1
            if n >= per_subset_cap:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v2/schema_first.jsonl")
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--per-subset", type=int, default=400, help="max distinct schemas taken from each JSONSchemaBench subset")
    ap.add_argument("--docs-per-schema", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="JSONSchemaBench split to take schemas from. SFT v2 used 'train'; the DPO pool uses 'val' and 'test' (schemas SFT never saw)")
    ap.add_argument("--id-prefix", default="ssf", help="row id prefix, so rows from another run can never collide with SFT rows")
    args = ap.parse_args()
    rng, fk = random.Random(args.seed), Faker("en_US")
    Faker.seed(args.seed)
    schemas = load_schemas(args.subsets, args.per_subset, args.split)
    out, stats = [], Counter()
    for subset, uid, base in schemas:
        for d in range(args.docs_per_schema):
            sch = nullable(base)
            absent_p = rng.uniform(0.15, 0.25)
            texts = {}
            gold = sample(sch, "root", (), fk, rng, texts, absent_p)
            style = rng.choice(["kv", "kv", "prose", "table"])
            doc = render(gold, texts, sch, rng, style, rng.choice(["human", "human", "title", "upper", "raw"]), sch.get("title", subset))
            if rng.random() < 0.35:
                doc = add_distractor(doc, gold, texts, rng)
            final = copy.deepcopy(sch)
            for p, (kind, _) in texts.items():  # tell the model what a date must look like
                if kind == "D":
                    sub = final
                    for x in p:
                        sub = sub["items"] if isinstance(x, int) else sub["properties"][x]
                    sub["format"] = "date"
                    if "YYYY-MM-DD" not in sub.get("description", ""):
                        sub["description"] = (sub.get("description", "") + " Format: YYYY-MM-DD.").strip()
            use_defs = rng.random() < 0.5
            if use_defs:
                final = lift_defs(final)
            try:
                jsonschema.Draft202012Validator.check_schema(final)
                jsonschema.Draft202012Validator(final).validate(gold)
            except Exception:
                stats["dropped_invalid"] += 1
                continue
            if not supported(gold, texts, doc) or len(doc) > 4200:
                stats["dropped_unsupported_or_long"] += 1
                continue
            stats["kept"] += 1
            stats[f"style:{style}"] += 1
            stats["with_defs"] += use_defs
            out.append({"id": f"{args.id_prefix}_{subset}_{uid}_{d}", "domain": f"schemastore/{subset}", "source": "schema_first:jsonschemabench",
                        "schema": final, "document": doc, "gold": gold, "schema_fp": fp(final),
                        "provenance": {"generator": "data/gen_schema_first.py", "schema_source": f"epfl-dlab/JSONSchemaBench/{subset}/{uid}",
                                       "style": style, "seed": args.seed, "defs": use_defs}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"{len(schemas)} schemas -> {len(out)} rows -> {args.out}")
    print(dict(stats))


if __name__ == "__main__":
    main()
