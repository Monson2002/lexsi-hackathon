# AGENTS.md

Context for any AI coding agent (Claude Code, Codex, Cursor, etc.) working in this repo. Read this fully before changing anything. If something here conflicts with what a human teammate tells you in the current session, follow the human and flag the conflict.

## Project in one paragraph

We are building a small fine-tuned Aya model for **schema-conditioned structured extraction**. Input: a document plus a JSON Schema. Output: one JSON object that validates against that schema and contains the correct values from the document. We are NOT training for one fixed schema. The goal is for the model to learn the general skill of following arbitrary schemas, and we prove it on schemas and a domain it never saw in training.

This is for the Lexsi Labs hackathon, Track 1 (Fine-tuning and Data Curation), problem statement 1 (structured output and extraction). Hackathon page: https://hackathon.lexsi.ai/

**Submission deadline: Nov 14, 2026, end of day (AoE).** Prefer boring, working, reproducible over clever. Anything that threatens the deadline gets cut.

## Hard rules from the hackathon (do not violate)

- The model we fine-tune must be a Cohere Labs open-weights model: Tiny Aya (Base / Global / Fire / Water / Earth) or Aya Expanse 8B / 32B. Default student: **Tiny Aya Global** (change only if the team says so).
- Other models are allowed only in a supporting role: teacher for synthetic data, validator, LLM-as-judge, or closed-model baseline for the cost comparison. Never swap the fine-tuned model for a non-Aya one.
- We must chain at least 2 of the track's toolkits, 3 preferred: **CuratorKIT** (data), **AlignTune** (training), **AuditKIT** (evaluation). Do not replace them with homegrown equivalents without asking. Custom code is fine where a toolkit has no hook (for example, extraction-specific metrics), but it should plug into the toolkit's flow.
- Everything is public on Hugging Face (models, adapters, datasets, logs) and the GitHub repo is public. Do not commit anything that cannot be public.
- The work must be ours, built during the hackathon. Do not paste in code or data of unclear license.
- No real patient, customer or personal data. Synthetic or public datasets only.

## What the model does (the contract)

**Prompt format is frozen once the team signs it off.** Training and evaluation must use the identical format. If you need to change it, stop and ask, because it invalidates data and results.

Placeholder until finalized (update this section when it is):

- Chat template of the Aya model.
- One block with the JSON Schema, one block with the document text.
- Instruction: return only a JSON object that conforms to the schema, no prose, no markdown fences.

**Output rules the model is trained to follow:**

- Valid JSON, validates against the given schema (types, enums, required fields, nesting, formats).
- Values come from the document. Never invent values.
- If a value is not in the document, output `null` (or omit the key only if the schema makes it optional and not nullable). Missing-field handling is a first-class requirement.
- Apply normalization only when the schema asks for it (for example date formats, numbers without currency symbols).

## Data approach

- **Gold is built schema-first, not document-first.** Generate a schema, sample a gold record that satisfies it, have a teacher write a realistic document containing exactly those values. Gold is then exact by construction. Do not use "LLM extracts from document" as gold.
- **Round-trip filter:** a strong model extracts from the generated document, and we drop examples where it disagrees with gold.
- **Automatic checks on every example:** JSON parses, validates with `jsonschema`, near-duplicate removal (documents and schemas), grounding check (extractive string values must appear in the document after normalization).
- **Diversity matters more than volume.** Vary schema depth, arrays of objects, enums, optional vs required, nullable fields, naming style, field count, document style (clean, messy, OCR-like noise, flattened tables), and length (keep training examples under about 3-4K tokens).
- **Deliberate hard cases:** missing fields (roughly 15-25% of fields), distractor values (two dates, two amounts), normalization, long arrays, fields that look related but are absent.
- Domains: invoices, support tickets, clinical-style notes (synthetic), forms/reports. Which one is held out is decided by the team; check the open questions in the team plan.

### Splits (never leak between them)

- `train`: training schemas and documents.
- `val`: held-out documents, seen schemas.
- `test_a`: held-out schemas, seen domains.
- `test_b`: held-out domain entirely.
- `test_c` (optional): small set of real public documents.

Split assignment happens at schema level and domain level, before generation or deduplication. A schema (or a near-duplicate of one) must never appear in more than one split. If you add data, rerun the leakage check.

### File format conventions (update when finalized)

- Datasets are JSONL, one example per line.
- Each record should carry at least: `id`, `domain`, `schema` (the JSON Schema), `document`, `gold` (the target JSON), and `split`. Add provenance fields (generator, teacher model, seed) so results are reproducible.
- Keep a fixed random seed in configs and log it.

## Training approach

- Stage 1: SFT with LoRA/QLoRA through AlignTune. Loss on the JSON output only. Starting point (tune from there): rank 16-64, lr around 1e-4 to 2e-4, 2-3 epochs, max length 4K.
- Stage 2 (optional, decided by about Nov 11): RL with a verifiable reward. Reward: 0 if not parseable, partial credit for schema validity, main term field-level F1 against gold **including correct nulls**, and a penalty for ungrounded (hallucinated) values.
- **Reward-hacking check is mandatory** before a long RL run. An all-null output is always schema-valid and sometimes right; the reward must not make it attractive, and the hallucination penalty must not push the model to null everything.
- SFT-only done well is a valid submission. Do not let RL block the deadline.

## Evaluation approach

Run all systems on the same test sets:

1. Base Tiny Aya, zero-shot, schema in the prompt.
2. Base plus constrained decoding.
3. Our SFT model.
4. Our SFT model plus constrained decoding.
5. Our SFT+RL model (if it exists).
6. Closed-model reference (supporting role, for the cost comparison).

Metrics (we report several; the rules require more than one):

- **Schema validity rate** (parses and validates). Note: constrained decoding makes this near 100% for any model, so it is not our headline.
- **Field-level F1 / exact match** against gold.
- **Document-level exact match** (all fields correct).
- **Hallucination rate** (extracted values not supported by the document).
- **Null accuracy** (correct handling of missing fields).
- **Generalization gap**: `test_a` and `test_b` vs `val`.
- **Cost and speed**: tokens/sec, GPU hours, estimated cost per 1,000 documents vs closed-API pricing.

Metric definitions must be implemented once, in one shared place (a module, or a utility file/dataset notebooks can import), and reused everywhere (training reward, eval, notebooks). Do not reimplement them per script, or numbers will not match across the write-up.

## Compute and workflow

- Most training and experimentation happens on **Kaggle** (notebooks, free GPUs) and **Hugging Face** (datasets, models, adapters, Spaces, credits). Assume Kaggle-style constraints: session time limits, limited GPU memory, no persistent disk between sessions, so checkpoint often and push artifacts to the Hugging Face Hub instead of relying on local files.
- Code should run top-to-bottom in a fresh Kaggle notebook: pin versions, read secrets (HF token) from Kaggle secrets or environment variables, and log the seed and config used.
- Save datasets and adapters to the HF Hub as soon as they are produced, with the split and generation details in the dataset/model card.

## Repo structure

**Not decided yet, on purpose.** We are building a prototype first and will organize the repo once we see what it looks like. Do not create a directory scaffold, reorganize, or move things into a "proper" layout on your own. Put new code where the person asking tells you to, or keep it in the existing notebook/script. When a structure is agreed, this section gets updated.

## How to work in this repo

- **Preserve existing function signatures and interfaces.** Teammates are building against them in parallel. If a signature genuinely has to change, say so in your summary and update every caller.
- **Keep structural changes minimal.** Prefer small, local edits over refactors or reorganizing files, unless asked. Do not rename or move files that other people may be using.
- **Deliver complete, working implementations** of what was asked rather than stubs or step-by-step partials. If something is blocked, say what and why instead of leaving silent TODOs.
- **Explain the mechanics** of non-obvious choices in comments or the summary (why this loss, why this reward term, why this split). The team needs to understand it for the write-up and Q&A, not just have it run.
- Do not install large dependencies, upgrade pinned versions, or change training hyperparameters/prompt format without saying so.
- Never fabricate results, citations or metrics. If a number was not produced by a run you can point to, do not write it down. If you cite a paper, only cite one you actually verified exists.
- Do not commit secrets (HF tokens, API keys). Use environment variables.
- You may read official Lexsi sources (docs sites and the Lexsi-Labs GitHub repos) whenever needed, and should do so rather than guess.
- Read the toolkit docs before using them. Do not guess APIs for CuratorKIT, AlignTune or AuditKIT:
  - CuratorKIT: https://curatorkit.lexsi.ai/latest/ and https://github.com/Lexsi-Labs/CuratorKIT
  - AlignTune: https://aligntune.lexsi.ai and https://github.com/Lexsi-Labs/aligntune
  - AuditKIT: https://auditkit.lexsi.ai and https://github.com/Lexsi-Labs/AuditKIT
- When you finish a task, summarize: what changed, what you ran and the result, what is still untested, and anything that affects other people's work.

## Deliverables checklist (what the repo must eventually contain)

- Schema and document generation code
- Curated dataset (train / val / test_a / test_b) on Hugging Face, public
- Raw generation logs on Hugging Face
- Adapters / merged model on Hugging Face, public, in a model collection
- Eval scripts and results tables (before vs after, multiple metrics)
- README with reproducibility notes
- Write-up (HF Article or README): metrics and why, approach, dataset and model rationale, results, limitations
- Cost comparison: closed model vs self-hosted fine-tuned model
- Public demo (an HF Space: paste a document and a schema, get JSON)
- Abstract under 500 words

## Reading (for context on why things are the way they are)

- Learning to Generate Structured Output with Schema Reinforcement Learning (Lu et al., ACL 2025): SchemaBench and RL with a schema validator. Closest to our plan.
- Tiny Aya: Bridging Scale and Multilingual Depth: https://arxiv.org/abs/2603.11510
- AlignTune: https://arxiv.org/pdf/2602.09621
- CuratorKIT: https://arxiv.org/pdf/2606.21631
- SO-Bench: https://arxiv.org/abs/2511.21750

The full reading list and the team plan live in the shared team doc.

## Things that will waste the team's time

- Reporting schema validity as the main result (constrained decoding fixes it for free).
- Letting schemas or near-duplicates leak across splits.
- Using LLM-extracted output as gold.
- Changing the prompt format mid-project.
- Long RL runs with an unchecked reward.
- Data that is uniform and easy, so the model looks great on val and fails on `test_a` / `test_b`.