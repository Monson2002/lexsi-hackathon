# Abstract

**Teaching a small multilingual model to follow any JSON Schema: a fine-tuning and evaluation study on Tiny Aya Global**

We fine-tune Tiny Aya Global (3.35B parameters) to read a document and a JSON Schema and return one JSON object, with `null` for any value that the document does not contain. We aim for the general skill of following arbitrary schemas, and we test it on schemas and a domain never seen in training.

**Method.** With CuratorKIT we built 2,760 training examples from three sources. The gold answer is exact by construction: we sample a record from the schema first and then write the document from it. Dates, amounts and enum labels appear in varied forms, and one to three fields are removed from the text, so the correct answer is `null`. A split guard removes any training schema that overlaps an evaluation set. With AlignTune we trained a QLoRA adapter on one free Kaggle T4 GPU, with the loss on the JSON answer only. With AuditKIT we evaluated on four splits: two in-domain splits, FieldBench `test_b` (a new domain, 415 documents) and `test_c` (143 real receipts). We report field-level F1, document exact match, null accuracy and hallucination rate. Schema validity is not a headline metric: constrained decoding makes it near 100% for any model.

**Results.** Against the base model, document exact match rises from 0.608 to 0.975 on `val` and from 0.533 to 0.958 on `test` (both significant). On the new domain the gain is small: field F1 rises from 0.795 to 0.815 (significant), but document exact match rises from 0.222 to 0.263 and the interval includes zero. On real receipts, field F1 rises from 0.447 to 0.494 (significant), but no system gets a single receipt fully correct. An earlier adapter scored 0.983 in domain and 0.010 on the new domain, because it had memorized the templates. More varied data removed that collapse.

**Negative results.** An earlier GRPO adapter matched the base model within about one point on every split. A DPO stage on top of our adapter (1,415 preference pairs, 247 of them the model's own mistakes) did not help: against the merged model that it started from, the change was within noise. We also found that merging the adapter into the weights costs about 3 points of document exact match on the new domain.

**Limits.** All results come from one run and one seed. Training documents are synthetic and template-like. We did not evaluate constrained decoding, and we did not run a closed-model baseline or a cost comparison. The adapter is slow on a T4 (about 0.04 documents per second). Part of the training data has an unresolved licence note, and the model is non-commercial like its base.

Code, evaluation scripts and verified result tables are on GitHub, and the adapters are on Hugging Face. The curated dataset is not yet published, because of the open licence question.
