# Model catalog

`internal/catalog/models.json` is the model catalog compiled into `ollama-compat`.
It is generated, not hand-edited:

```bash
go run ./scripts/gen-catalog -in catalog/sources.yaml -out internal/catalog/models.json
```

`catalog/sources.yaml` is the hand-maintained input: family, display name,
`params_b`, `quality`, `quality_source`, optional `kv_quant_ok` (default true)
and a list of Ollama library tags. The generator is a dev-only tool that needs
network access to `registry.ollama.ai`; the shipped binary never runs it and
makes no network calls for the catalog.

## Where each field comes from

| Field | Source |
|---|---|
| `family`, `name`, `params_b` | `sources.yaml` (`params_b` from the model card) |
| `quality` | `sources.yaml`, see "Quality scores" below |
| `variants[].size_bytes` | registry manifest, size of the `application/vnd.ollama.image.model` layer |
| `variants[].bits` | `size_bytes * 8 / (params_b * 1e9)`, rounded to 2 decimals |
| `variants[].quant` | tag suffix (for example `q4_K_M`); if the tag has none, the GGUF `general.file_type` |
| `layers` | GGUF `<arch>.block_count` |
| `kv_heads` | GGUF `<arch>.attention.head_count_kv` |
| `head_dim` | GGUF `<arch>.attention.key_length`, else `embedding_length / head_count` |
| `max_ctx` | GGUF `<arch>.context_length` (the trained maximum, as declared in the file) |
| `active_params_b` | `sources.yaml` `active_params_b` when set (model card value; set for every MoE entry), else for mixture-of-experts models `params_b * expert_used_count / expert_count`, else `params_b` |

The GGUF fields are read from the first 8 MiB of the model blob with an HTTP
range request (64 MiB retry if the tokenizer arrays push the needed keys
further). All architecture fields come from the first tag of each entry; the
quants of one model share an architecture.

### Known approximations

- **MoE active parameters.** Every MoE entry sets `active_params_b` in
  `sources.yaml` from its model card: Qwen3 30B-A3B 3.3B, Mixtral 8x7B 12.9B,
  Granite 3.1 MoE 3B 0.8B. The fallback formula
  `params_b * expert_used_count / expert_count` ignores the attention and
  embedding weights every token uses, so it understates active parameters
  (Qwen3 30B-A3B would come out at 1.91B). It applies only to a new MoE entry
  added without the override; set the field when adding one.
- **`bits` uses the model-card parameter count.** Small models with large
  vocabularies (Qwen3 0.6B/1.7B, Gemma 3 1B, Gemma 2 2B) show high bits per
  weight because embedding and output tensors are kept at higher precision or
  stored untied. Gemma 3 4B/12B/27B show slightly low bits because the card
  count includes the vision tower. `size_bytes` is exact either way.
- **`max_ctx`** is the file's declared limit; Mistral NeMo declares 1,024,000
  although it is advertised for 128K.

## Quality scores

Scores are **maintainer-curated estimates informed by public benchmarks
(source not machine-fetched), as of 2026-11-16.** No single public aggregate
leaderboard covers this whole set (small and older models are missing from
current arena-style boards, and newer families are missing from the retired
Open LLM Leaderboard), so no leaderboard was fetched and no citation is given.

Scale: cross-family 0-100. The best catalog model (Qwen3 32B) is 90 and a 1B
model (Llama 3.2 1B Instruct) is 30. Scores describe the unquantized model;
the picker subtracts the quantization penalty (`catalog.QuantPenalty`:
q8 0, q6 -1, q5 -2, q4 -3, q3 -8, q2 -15). The picker's `quality >= 40` floor
is defined on this scale. A model with quality 0 is unscored and is excluded
from picks.

## Registry notes (2026-11-16)

These tags return 404 and are left out of `sources.yaml`; the entry carries
`q8_0` as its second quant instead: `qwen3:32b-q3_K_M`,
`qwen3:30b-a3b-q3_K_M`, `gemma3:27b-it-q3_K_M`,
`deepseek-r1:32b-qwen-distill-q3_K_M`, `deepseek-r1:70b-llama-distill-q3_K_M`.
`granite3.3` publishes only size tags (`granite3.3:2b`, `granite3.3:8b`, both
q4_K_M), so those entries have one variant.

The generator drops any tag that 404s and logs `DROP tag <tag>: 404`.
