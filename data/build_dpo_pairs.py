"""Build the DPO preference pairs (docs/DPO_PLAN.md WP1 to WP3).

    python data/build_dpo_pairs.py prompts                       # 1) fresh prompt pool, guarded -> data/dpo/pool_clean.jsonl
    python data/build_dpo_pairs.py build                         # 2a) rule-based pairs only
    python data/build_dpo_pairs.py build --onpolicy samples.jsonl   # 2b) rule-based + the SFT v2 model's own failures

Pair = (prompt, chosen, rejected). Prompt = the frozen prompt (system + <JSON_SCHEMA>/<DOCUMENT> user turn). Chosen = the gold JSON, written exactly as in
SFT (json.dumps, sorted keys). Rejected = an answer that is WORSE under the shared metric (checked for every pair), either
  - rule-based: the gold with exactly one injected failure of a kind SFT/base models actually make (dates left as written, required key dropped, extra key,
    schema echoed, invented value, value turned into null, enum label outside the list, wrong value, wrong type, truncated JSON, ``` fence), or
  - on-policy: a real answer sampled from the SFT v2 model (data/sample_onpolicy.py) that scores below the gold.
Both directions of the null error are included (invented value AND value turned into null) so the model cannot learn "write more nulls".

Guards (AGENTS.md: a schema must never be in two splits): prompts whose schema or field-name set also appears in SFT v2 training (the model has seen them, so its
mistakes there are not representative) or in val / test / test_b / test_c are dropped BEFORE any pair is built. Evaluation sets are never a pair source.
CuratorKIT does the curation (schema/length gate, MinHash dedup, secrets gate, provenance). Its dpo export is NOT used because it keeps only the
instruction field and drops the document; the final file is written here from the surviving ids, as chat messages.
"""
import argparse
import copy
import importlib.util
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import leakage_check as lc  # noqa: E402

DPO_DIR = HERE / "dpo"
SYSTEM_PROMPT = (
    "You extract structured data. Return only a single JSON object that conforms to the given JSON Schema. "
    "Use null for values that are not present in the document. Do not invent values. "
    "Do not add explanations or markdown fences."
)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# Starting mix of rule-based kinds. Replace with the SFT v2 failure counts (error_taxonomy.py) via --weights once they exist.
DEFAULT_WEIGHTS = {"date_unnormalised": 0.18, "missing_key": 0.15, "invented_value": 0.13, "over_null": 0.10, "wrong_value": 0.10, "extra_key": 0.08,
                   "schema_echo": 0.08, "enum_violation": 0.08, "wrong_type": 0.04, "truncated": 0.03, "fenced": 0.03}


def load_eval_module():
    spec = importlib.util.spec_from_file_location("ev", str(HERE.parent / "eval.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def load(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def user_message(schema, document):
    return f"<JSON_SCHEMA>\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n</JSON_SCHEMA>\n\n<DOCUMENT>\n{document}\n</DOCUMENT>"


def dump(gold):
    return json.dumps(gold, ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------------ gold helpers (paths are tuples of keys and list indices)
def walk(g, path=()):
    if isinstance(g, dict):
        for k, v in g.items():
            yield from walk(v, path + (k,))
    elif isinstance(g, list):
        for i, v in enumerate(g):
            yield from walk(v, path + (i,))
    else:
        yield path, g


def get(g, path):
    for p in path:
        g = g[p]
    return g


def put(g, path, v):
    for p in path[:-1]:
        g = g[p]
    g[path[-1]] = v


def schema_at(schema, path, root=None):
    """Sub-schema for a gold path, following $ref and anyOf-with-null; None if it cannot be resolved."""
    root = root or schema
    node = schema
    for p in path:
        node = deref(node, root)
        if not isinstance(node, dict):
            return None
        node = node["items"] if isinstance(p, int) else (node.get("properties") or {}).get(p)
        if node is None:
            return None
    return deref(node, root)


def deref(node, root):
    for _ in range(8):
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            cur = root
            for part in node["$ref"][2:].split("/"):
                cur = cur[part]
            node = cur
        elif "anyOf" in node:
            nonnull = [b for b in node["anyOf"] if b.get("type") != "null"]
            node = nonnull[0] if nonnull else node
            if not nonnull:
                return node
        else:
            return node
    return node


def date_surfaces(iso):
    """Every way the generators write an ISO date in a document."""
    y, m, d = map(int, iso.split("-"))
    out = [f"{MONTHS[m - 1]} {d}, {y}", f"{d} {MONTHS[m - 1]} {y}", f"{MONTHS[m - 1][:3]} {d}, {y}", f"{d:02d}-{MONTHS[m - 1][:3]}-{y}"]
    if d > 12:
        out += [f"{d:02d}/{m:02d}/{y}", f"{m:02d}/{d:02d}/{y}", f"{d:02d}.{m:02d}.{y}"]
    return out


# ------------------------------------------------------------------ rule-based corruptions: each returns the rejected TEXT or None if not applicable
def c_date_unnormalised(row, rng):
    iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    cand = [(p, v) for p, v in walk(row["gold"]) if isinstance(v, str) and iso.match(v)]
    rng.shuffle(cand)
    for p, v in cand:
        found = [s for s in date_surfaces(v) if s in row["document"]]
        if found:
            g = copy.deepcopy(row["gold"])
            put(g, p, found[0])
            return dump(g)
    return None


def c_missing_key(row, rng):
    keys = [(p, v) for p, v in walk(row["gold"]) if p and isinstance(p[-1], str)]
    nonnull = [p for p, v in keys if v is not None] or [p for p, _ in keys]
    if not nonnull:
        return None
    p = rng.choice(nonnull)
    g = copy.deepcopy(row["gold"])
    del get(g, p[:-1])[p[-1]]
    return dump(g)


def c_extra_key(row, rng):
    g = copy.deepcopy(row["gold"])
    names = [k for k in g if isinstance(k, str)] or ["field"]
    base = rng.choice(names)
    new = rng.choice([f"{base}_code", f"{base}_id", f"{base}_text", "extra_field", "notes"])
    if new in g:
        return None
    g[new] = rng.choice(["N/A", "unknown", "", "see document"])
    return dump(g)


def c_schema_echo(row, rng):
    g = copy.deepcopy(row["gold"])
    if rng.random() < 0.5:  # a value replaced by its schema fragment (seen in SFT on real documents)
        leaves = [p for p, v in walk(g) if isinstance(p[-1], str)]
        if not leaves:
            return None
        p = rng.choice(leaves)
        put(g, p, {"type": rng.choice(["string", "number", "object"])})
        return dump(g)
    props = list((row["schema"].get("properties") or {}).items())[:4]  # the schema itself as the answer
    if not props:
        return None
    return dump({"type": "object", "properties": {k: {"type": "string"} for k, _ in props}, "required": [k for k, _ in props]})


def c_invented_value(row, rng):
    nulls = [p for p, v in walk(row["gold"]) if v is None]
    strings = [v for _, v in walk(row["gold"]) if isinstance(v, str) and len(v) > 2]
    if not nulls or not strings:
        return None
    p = rng.choice(nulls)
    g = copy.deepcopy(row["gold"])
    put(g, p, rng.choice(strings))  # a value that belongs to another field: the typical "filled the empty slot" mistake
    return dump(g)


def c_over_null(row, rng):
    cand = [p for p, v in walk(row["gold"]) if v is not None and not isinstance(v, bool) and not any(isinstance(x, int) for x in p)]
    if not cand:
        return None
    g = copy.deepcopy(row["gold"])
    put(g, rng.choice(cand), None)
    return dump(g)


def c_wrong_value(row, rng):
    strings = [(p, v) for p, v in walk(row["gold"]) if isinstance(v, str) and len(v) > 2]
    if len(strings) < 2:
        return None
    (p1, v1), (p2, v2) = rng.sample(strings, 2)
    if v1 == v2:
        return None
    g = copy.deepcopy(row["gold"])
    put(g, p1, v2)  # two fields swapped content: the "picked the wrong date/name" mistake
    return dump(g)


def c_enum_violation(row, rng):
    cand = []
    for p, v in walk(row["gold"]):
        sub = schema_at(row["schema"], p)
        if isinstance(sub, dict) and sub.get("enum") and isinstance(v, str):
            cand.append((p, v))
    if not cand:
        return None
    p, v = rng.choice(cand)
    g = copy.deepcopy(row["gold"])
    put(g, p, rng.choice([f"{v} Agreement", v.lower() + " (other)", "Other", v.replace("_", " ").title() + "s"]))
    return dump(g)


def c_wrong_type(row, rng):
    nums = [p for p, v in walk(row["gold"]) if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return None
    g = copy.deepcopy(row["gold"])
    p = rng.choice(nums)
    put(g, p, f"{get(g, p)}")
    return dump(g)


def c_truncated(row, rng):
    t = dump(row["gold"])
    return t[: int(len(t) * rng.uniform(0.55, 0.85))] if len(t) > 40 else None


def c_fenced(row, rng):
    return "```json\n" + dump(row["gold"]) + "\n```"


KINDS = {"date_unnormalised": c_date_unnormalised, "missing_key": c_missing_key, "extra_key": c_extra_key, "schema_echo": c_schema_echo,
         "invented_value": c_invented_value, "over_null": c_over_null, "wrong_value": c_wrong_value, "enum_violation": c_enum_violation,
         "wrong_type": c_wrong_type, "truncated": c_truncated, "fenced": c_fenced}


def is_worse(ev, rejected_text, row, kind):
    """Rejected must score strictly below the gold under the shared metric. A fence is the one exception (our parser accepts it, the prompt forbids it)."""
    if kind == "fenced":
        return rejected_text.startswith("```") and ev.parse_json_object(rejected_text)[0]
    m = ev.compute_field_metrics(rejected_text, row["gold"], row["schema"])
    return m["doc_exact"] == 0.0


# ------------------------------------------------------------------ step 1: the guarded fresh pool
def guard_sets(args):
    sft = load(HERE / "sft_v2" / "train_v2.jsonl")
    evals = load(HERE / "aya-json-sft-dataset" / "val.jsonl") + load(HERE / "aya-json-sft-dataset" / "test.jsonl")
    for p in args.eval_extra:
        evals += load(p)
    return ({lc.canon(r["schema"]) for r in sft}, {lc.field_names(r["schema"]) for r in sft},
            {lc.canon(r["schema"]) for r in evals}, {lc.field_names(r["schema"]) for r in evals}, {r["id"] for r in sft})


def cmd_prompts(args):
    pools = [row for f in sorted(DPO_DIR.glob("pool_*.jsonl")) if f.name != "pool_clean.jsonl" for row in load(f)]
    sc, sn, ec, en, sft_ids = guard_sets(args)
    keep, drop = [], Counter()
    for r in pools:
        c, n = lc.canon(r["schema"]), lc.field_names(r["schema"])
        if r["id"] in sft_ids:
            drop["id collides with an SFT row"] += 1
        elif c in ec or n in en:
            drop["schema or field-name set also in val/test/FieldBench"] += 1
        elif c in sc or n in sn:
            drop["schema or field-name set seen in SFT v2 training"] += 1
        else:
            keep.append(r)
    with open(DPO_DIR / "pool_clean.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pool rows {len(pools)} -> {len(keep)} kept; dropped: {dict(drop)}; distinct schemas kept: {len({lc.canon(r['schema']) for r in keep})}")


# ------------------------------------------------------------------ step 2: pairs
def cmd_build(args):
    ev = load_eval_module()
    rng = random.Random(args.seed)
    pool = load(DPO_DIR / "pool_clean.jsonl")
    outdir = Path(args.out_dir) if args.out_dir else DPO_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in pool}
    weights = json.loads(Path(args.weights).read_text()) if args.weights else DEFAULT_WEIGHTS
    pairs, stats = [], Counter()

    # on-policy pairs: the model's own wrong answers (one per prompt: the lowest field_f1)
    if args.onpolicy:
        best = {}
        for s in load(args.onpolicy):
            r = by_id.get(s["id"])
            if r is None:
                stats["onpolicy: id not in the clean pool"] += 1
                continue
            m = ev.compute_field_metrics(s["raw_output"], r["gold"], r["schema"])
            if m["doc_exact"] == 1.0 or s["raw_output"].strip() == dump(r["gold"]):
                stats["onpolicy: answer already correct (no pair)"] += 1
                continue
            if r["id"] not in best or m["field_f1"] < best[r["id"]][1]:
                best[r["id"]] = (s["raw_output"].strip(), m["field_f1"])
        for i, (txt, f1) in best.items():
            r = by_id[i]
            pairs.append({"pair_id": f"{i}::onpolicy", "prompt_id": i, "kind": "onpolicy", "source": "sft_v2_samples", "chosen": dump(r["gold"]), "rejected": txt})
        stats["pairs: on-policy"] = len(best)

    # rule-based, with QUOTAS per kind. Drawing prompts at random starves the kinds that apply to few prompts (dates need a date in the document,
    # enums need an enum field), yet unnormalised dates were the base model's biggest failure. So: compute what applies to every prompt, give each kind
    # a quota from the weights, fill the scarcest kinds first, then top up from whatever is left.
    app = {}  # prompt_id -> {kind: rejected text}
    for r in pool:
        for k, fn in KINDS.items():
            if weights.get(k, 0) > 0:
                t = fn(r, rng)
                if t is not None and t != dump(r["gold"]) and is_worse(ev, t, r, k):
                    app.setdefault(r["id"], {})[k] = t
    total = args.n_rule or sum(min(args.per_prompt, len(v)) for v in app.values())
    wsum = sum(weights[k] for k in KINDS if weights.get(k, 0) > 0)
    quota = {k: round(total * weights[k] / wsum) for k in KINDS if weights.get(k, 0) > 0}
    avail = {k: [i for i, v in app.items() if k in v] for k in quota}
    used, picked = Counter(), set()
    n_rule = 0

    def take(k, i):
        nonlocal n_rule
        r = by_id[i]
        pairs.append({"pair_id": f"{i}::{k}", "prompt_id": i, "kind": k, "source": "rule_based", "chosen": dump(r["gold"]), "rejected": app[i][k]})
        used[i] += 1
        picked.add((i, k))
        n_rule += 1

    for k in sorted(quota, key=lambda x: len(avail[x])):  # scarcest first
        cand = [i for i in avail[k] if used[i] < args.per_prompt]
        rng.shuffle(cand)
        for i in cand[: quota[k]]:
            take(k, i)
        if len(avail[k]) < quota[k]:
            stats[f"quota short: {k} (wanted {quota[k]}, only {len(avail[k])} prompts allow it)"] = 1
    rest = [(i, k) for k in quota for i in avail[k] if (i, k) not in picked and used[i] < args.per_prompt]
    rng.shuffle(rest)
    while rest and n_rule < total:
        i, k = rest.pop()
        if used[i] < args.per_prompt and (i, k) not in picked:
            take(k, i)
    stats["pairs: rule-based"] = n_rule
    for k, v in Counter(p["kind"] for p in pairs).items():
        stats[f"  kind {k}"] = v

    # ---- CuratorKIT: schema/length gate, MinHash dedup, secrets gate, provenance
    ck_in = outdir / "curatorkit_input.jsonl"
    with open(ck_in, "w", encoding="utf-8") as f:
        for p in pairs:
            r = by_id[p["prompt_id"]]
            f.write(json.dumps({"id": p["pair_id"], "instruction": SYSTEM_PROMPT, "input": user_message(r["schema"], r["document"]), "output": p["chosen"],
                                "chosen": p["chosen"], "rejected": p["rejected"], "task_type": "preference",
                                "metadata": {"kind": p["kind"], "source": p["source"]}}) + "\n")
    from curatorkit import Curator, CuratorConfig
    ck_dir = outdir / "curatorkit"
    if ck_dir.exists():
        shutil.rmtree(ck_dir)
    res = Curator(CuratorConfig(dataset=str(ck_in), dedup=args.dedup, minhash_threshold=0.85, clean=False, schema_gate=True, min_tokens=10, max_tokens=6000,
                                secrets_gate=True, export_formats=["dpo"], output_dir=str(ck_dir), name="dpo_pairs")).run()
    kept = {s.id for s in res.passed}
    manifest = json.load(open(ck_dir / "manifest.json"))
    stats["curatorkit: in"] = len(pairs)
    stats["curatorkit: out"] = len(kept)
    stats["curatorkit: rejected"] = json.dumps(manifest.get("rejected_breakdown"))
    stats["curatorkit: near-duplicates removed"] = manifest.get("dedup_stats", {}).get("near_duplicates_removed")

    # ---- our own final write: length cap, final sanity, chat-message format
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
    except Exception as e:  # no access to the gated tokenizer: fall back to the character estimate used for SFT
        stats["note"] = f"tokenizer unavailable ({type(e).__name__}); lengths estimated at 3.2 chars/token"
    final = []
    for p in pairs:
        if p["pair_id"] not in kept:
            continue
        r = by_id[p["prompt_id"]]
        u = user_message(r["schema"], r["document"])
        # DPO feeds prompt+chosen AND prompt+rejected: the longer of the two must fit (an on-policy rejected answer can be longer than the gold)
        n = max(len(tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u}, {"role": "assistant", "content": a}],
                                            tokenize=True, return_dict=True)["input_ids"]) if tok else (len(SYSTEM_PROMPT) + len(u) + len(a)) / 3.2
                for a in (p["chosen"], p["rejected"]))
        if n > args.max_seq:
            stats["dropped: over the length cap"] += 1
            continue
        final.append({"id": p["pair_id"], "kind": p["kind"], "source": p["source"], "prompt_id": p["prompt_id"], "tokens_prompt_plus_chosen": int(n),
                      "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u}],
                      "chosen": [{"role": "assistant", "content": p["chosen"]}], "rejected": [{"role": "assistant", "content": p["rejected"]}]})
    out = outdir / "dpo_pairs_train.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in final:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    stats["final pairs"] = len(final)
    stats["final distinct prompts"] = len({p["prompt_id"] for p in final})
    lines = ["# DPO pairs build statistics", "", *(f"- {k}: {v}" for k, v in stats.items())]
    (outdir / "STATS_dpo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prompts", "build"):
        sp = sub.add_parser(name)
        sp.add_argument("--eval-extra", nargs="*", default=[], help="FieldBench test_b / test_c files, for the guard")
        sp.add_argument("--seed", type=int, default=42)
        if name == "build":
            sp.add_argument("--onpolicy", help="JSONL from data/sample_onpolicy.py: {id, raw_output}")
            sp.add_argument("--out-dir", help="where to write the pairs, stats and CuratorKIT files (default data/dpo); use another folder for tests")
            sp.add_argument("--weights", help="JSON {kind: weight} to replace the default mix")
            sp.add_argument("--n-rule", type=int, default=0, help="stop after this many rule-based pairs (0 = all prompts)")
            sp.add_argument("--per-prompt", type=int, default=2, help="max rule-based pairs per prompt")
            sp.add_argument("--dedup", default="exact", choices=["exact", "minhash"], help="CuratorKIT dedup; minhash removes sibling pairs that share a prompt")
            sp.add_argument("--max-seq", type=int, default=2048, help="max tokens of prompt + chosen answer")
    args = ap.parse_args()
    {"prompts": cmd_prompts, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
