# SFT v2 build statistics

- source dataset revision: sujalgawas/aya-json-sft-dataset @ e26e691b2c4c2f71f10f8df403ac8da4b0af4b84 (2026-11-10 14:29:13+00:00)
- sources pooled: 2994 (augmented 1517, schema-first 1232, paraloq 220, unaugmented originals 25)
- CuratorKIT 0.1.24: 2994 -> 2832; rejected by step/reason: {'secret_detected:Hex High Entropy String': 17, 'secret_detected:Base64 High Entropy String': 4, 'secret_detected:Base64 High Entropy String,Hex High Entropy String': 1}; MinHash near-duplicates removed: {'near_duplicates_removed': 140, 'surviving_count': 2854}
- jsonschema re-check on survivors: 0 dropped; over the length cap (2300 est. tokens): 72 dropped; split guard (schema or field-name set also in val/test/FieldBench): 0 dropped
- final rows: 2760; by source: {'augmented': 1411, 'schema_first': 1176, 'paraloq/json_data_extraction': 151, 'instruct_teacher': 8, 'natural_domain_template': 14}
- null leaves: 5079/30100 = 16.9%
- ISO-date gold values: 2158; appearing verbatim (ISO) in the document: 333 (15%)
- rows with $defs/$ref: 337
- distinct domains: 20; distinct schemas (fp): 2757
- median document chars: 308

## Leakage vs evaluation sets

- aya-json-sft val (120 rows): same schema 0, same field-name set 0, near-dup field names (J>=.8) 0, same document 0; gold string values also in train gold: 79.6%
- aya-json-sft test (120 rows): same schema 0, same field-name set 0, near-dup field names (J>=.8) 0, same document 0; gold string values also in train gold: 78.7%
- test_b.jsonl (415 rows): same schema 0, same field-name set 0, near-dup field names (J>=.8) 0, same document 0; gold string values also in train gold: 6.3%
- test_c.jsonl (143 rows): same schema 0, same field-name set 0, near-dup field names (J>=.8) 0, same document 0; gold string values also in train gold: 1.6%
