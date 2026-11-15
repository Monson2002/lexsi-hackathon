"""Build the SFT v2 training set: merge the three sources, curate with CuratorKIT, verify, export in the frozen prompt format.

    python data/augment_train.py            # -> data/sft_v2/augmented.jsonl     (fixes verbatim dates/numbers/enums, adds nulls)
    python data/gen_schema_first.py         # -> data/sft_v2/schema_first.jsonl  (JSONSchemaBench structure, $defs, nesting)
    python data/convert_paraloq.py          # -> data/sft_v2/paraloq.jsonl       (484 -> ~220 rows, $defs, real nesting)
    python data/build_sft_v2.py             # -> data/sft_v2/train_v2.jsonl, sft_messages_train.jsonl, STATS.md, curatorkit/

CuratorKIT does: JSONL ingest, SchemaGate (non-empty + token-length), exact + MinHash near-dup removal (0.85), SecretsGate,
and the audit trail (manifest.json, rejected.jsonl, lexsi_provenance.json, checksums). Choices that differ from its defaults:
  - clean=False: its text cleaner rewrites whitespace/unicode, which could change a document under a gold value
  - its 'messages' export is NOT used: it puts the system prompt inside the user turn, and the prompt format is frozen
    (system role + <JSON_SCHEMA>/<DOCUMENT> user turn + assistant JSON, as in the SFT notebook)
  - PII pseudonymisation is off: all data is synthetic, and replacing values would break exact gold
Val/test (aya-json-sft-dataset) and FieldBench are never training sources; STATS.md reports the leakage check against them.
"""
import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).parent))
import leakage_check as lc  # noqa: E402

# Frozen prompt (identical to eval.py and the SFT notebook, section 5)
SYSTEM_PROMPT = (
    "You extract structured data. Return only a single JSON object that conforms to the given JSON Schema. "
    "Use null for values that are not present in the document. Do not invent values. "
    "Do not add explanations or markdown fences."
)


def user_message(schema, document):
    return f"<JSON_SCHEMA>\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n</JSON_SCHEMA>\n\n<DOCUMENT>\n{document}\n</DOCUMENT>"


def load(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    else:
        yield o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/sft_v2")
    ap.add_argument("--orig", default="data/aya-json-sft-dataset")
    ap.add_argument("--eval-extra", nargs="*", default=[], help="extra eval JSONL files (e.g. FieldBench test_b/test_c) for the leakage report")
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--max-est-tokens", type=int, default=2300,
                    help="drop rows whose system+user+assistant text is estimated above this (3.2 chars/token; the real tokenizer is gated). "
                         "Eval truncates prompts at 2048 tokens and a cut-off answer teaches nothing")
    args = ap.parse_args()
    d = Path(args.dir)

    aug, ssf, par = load(d / "augmented.jsonl"), load(d / "schema_first.jsonl"), load(d / "paraloq.jsonl")
    # Originals only where augmentation did not apply (the rest are superseded by their augmented copy)
    done = {r["provenance"]["base_id"] for r in aug}
    orig = [r for r in load(Path(args.orig) / "train.jsonl") if r["id"] not in done]
    pool = par + ssf + aug + orig  # MinHash keeps the first of a near-duplicate pair
    by_id = {r["id"]: r for r in pool}
    assert len(by_id) == len(pool), "duplicate ids across sources"

    # ---- CuratorKIT
    ck_in = d / "curatorkit_input.jsonl"
    with open(ck_in, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps({"id": r["id"], "instruction": SYSTEM_PROMPT, "input": user_message(r["schema"], r["document"]),
                                "output": json.dumps(r["gold"], ensure_ascii=False, sort_keys=True), "task_type": "instruction_following",
                                "metadata": {"domain": r["domain"], "source": r["source"]}}) + "\n")
    from curatorkit import Curator, CuratorConfig
    ck_dir = d / "curatorkit"
    if ck_dir.exists():
        shutil.rmtree(ck_dir)
    cfg = CuratorConfig(dataset=str(ck_in), dedup="minhash", minhash_threshold=0.85, clean=False, schema_gate=True, min_tokens=10,
                        max_tokens=args.max_tokens, secrets_gate=True, export_formats=["alpaca"], output_dir=str(ck_dir), name="sft_v2")
    res = Curator(cfg).run()
    kept_ids = [s.id for s in res.passed]
    manifest = json.load(open(ck_dir / "manifest.json"))
    print("CuratorKIT:", len(pool), "->", len(kept_ids), "| rejected:", manifest.get("rejected_breakdown"), "| near-dups removed:", manifest.get("dedup_stats"))

    # ---- our own verification on the survivors (jsonschema again; ids must exist)
    # Split guard (AGENTS.md: a schema must never appear in more than one split). Sujal's dataset changed under us once: rows that were
    # in an older train were moved into the newer val/test, so training rows built from the old train shared schemas with them.
    # Drop every training row whose canonical schema or field-name set equals one in val, test or the extra eval files.
    guard_rows = load(Path(args.orig) / "val.jsonl") + load(Path(args.orig) / "test.jsonl") + [x for p in args.eval_extra for x in load(p)]
    block_canon = {lc.canon(x["schema"]) for x in guard_rows}
    block_names = {lc.field_names(x["schema"]) for x in guard_rows}
    final, bad, too_long, guarded = [], 0, 0, 0
    for i in kept_ids:
        r = by_id[i]
        if lc.canon(r["schema"]) in block_canon or lc.field_names(r["schema"]) in block_names:
            guarded += 1
            continue
        if (len(SYSTEM_PROMPT) + len(user_message(r["schema"], r["document"])) + len(json.dumps(r["gold"], ensure_ascii=False, sort_keys=True))) / 3.2 > args.max_est_tokens:
            too_long += 1
            continue
        try:
            jsonschema.Draft202012Validator(r["schema"]).validate(r["gold"])
        except jsonschema.ValidationError:
            bad += 1
            continue
        final.append(r)
    random.Random(42).shuffle(final)
    with open(d / "train_v2.jsonl", "w", encoding="utf-8") as f, open(d / "sft_messages_train.jsonl", "w", encoding="utf-8") as g:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            g.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                             {"role": "user", "content": user_message(r["schema"], r["document"])},
                                             {"role": "assistant", "content": json.dumps(r["gold"], ensure_ascii=False, sort_keys=True)}]},
                               ensure_ascii=False) + "\n")

    # ---- statistics
    tot = sum(1 for r in final for _ in leaves(r["gold"]))
    nul = sum(v is None for r in final for v in leaves(r["gold"]))
    dts = [(r, v) for r in final for v in leaves(r["gold"]) if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)]
    src = Counter(r["source"].split(":")[0] for r in final)
    rev = (Path(args.orig) / "REVISION.txt").read_text().strip() if (Path(args.orig) / "REVISION.txt").exists() else "unknown (no REVISION.txt)"
    lines = [f"# SFT v2 build statistics", "", f"- source dataset revision: {rev}",
             f"- sources pooled: {len(pool)} (augmented {len(aug)}, schema-first {len(ssf)}, paraloq {len(par)}, unaugmented originals {len(orig)})",
             f"- CuratorKIT 0.1.24: {len(pool)} -> {len(kept_ids)}; rejected by step/reason: {manifest.get('rejected_breakdown')}; MinHash near-duplicates removed: {manifest.get('dedup_stats')}",
             f"- jsonschema re-check on survivors: {bad} dropped; over the length cap ({args.max_est_tokens} est. tokens): {too_long} dropped; split guard (schema or field-name set also in val/test/FieldBench): {guarded} dropped",
             f"- final rows: {len(final)}; by source: {dict(src)}",
             f"- null leaves: {nul}/{tot} = {nul / tot:.1%}",
             f"- ISO-date gold values: {len(dts)}; appearing verbatim (ISO) in the document: {sum(v in r['document'] for r, v in dts)} ({sum(v in r['document'] for r, v in dts) / max(len(dts), 1):.0%})",
             f"- rows with $defs/$ref: {sum('$defs' in json.dumps(r['schema']) for r in final)}",
             f"- distinct domains: {len({r['domain'] for r in final})}; distinct schemas (fp): {len({r['schema_fp'] for r in final})}",
             f"- median document chars: {sorted(len(r['document']) for r in final)[len(final) // 2]}", "", "## Leakage vs evaluation sets", ""]

    # ---- leakage vs the evaluation sets
    evals = {"aya-json-sft val": load(Path(args.orig) / "val.jsonl"), "aya-json-sft test": load(Path(args.orig) / "test.jsonl")}
    for p in args.eval_extra:
        evals[Path(p).name] = load(p)
    seen_vals = {v for r in final for v in leaves(r["gold"]) if isinstance(v, str)}
    train_canon = {lc.canon(r["schema"]) for r in final}
    train_names = [lc.field_names(r["schema"]) for r in final]
    train_docs = {r["document"] for r in final}
    for name, rows in evals.items():
        same_schema = sum(lc.canon(r["schema"]) in train_canon for r in rows)
        same_names = sum(lc.field_names(r["schema"]) in set(train_names) for r in rows)
        near = sum(any(len(fx & f) / max(len(fx | f), 1) >= 0.8 for f in train_names) for fx in (lc.field_names(r["schema"]) for r in rows))
        same_doc = sum(r["document"] in train_docs for r in rows)
        vals = [v for r in rows for v in leaves(r["gold"]) if isinstance(v, str)]
        lines.append(f"- {name} ({len(rows)} rows): same schema {same_schema}, same field-name set {same_names}, near-dup field names (J>=.8) {near}, same document {same_doc}; "
                     f"gold string values also in train gold: {sum(v in seen_vals for v in vals) / max(len(vals), 1):.1%}")
    (d / "STATS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
