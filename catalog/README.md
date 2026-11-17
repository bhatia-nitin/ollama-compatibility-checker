# Model catalog

`internal/catalog/models.json` is the model catalog compiled into `ollama-compat`.
It is generated, not hand-edited:

```bash
go run ./scripts/gen-catalog -in catalog/sources.yaml -out internal/catalog/models.json
```

`catalog/sources.yaml` is the hand-maintained input: family, display name,
`params_b`, `quality`, `quality_source`, optional `kv_quant_ok` (default true),
optional `tasks` and a list of Ollama library tags. The generator is a dev-only tool that needs
network access to `registry.ollama.ai`; the shipped binary never runs it and
makes no network calls for the catalog.

## Where each field comes from

| Field | Source |
|---|---|
| `family`, `name`, `params_b` | `sources.yaml` (`params_b` from the model card) |
| `quality` | `sources.yaml`, see "Quality scores" below |
| `tasks` | `sources.yaml`: `coding` for code-tuned models, `reasoning` for models with a thinking mode, `vision` for models that accept images in Ollama. Facts from the model card, not quality judgments |
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

## Calibration log

Each entry records one `ollama-compat bench` run against the estimate from
`recommend` on the same machine. Add new entries at the bottom.

| Field | Value |
|---|---|
| Date | 2026-11-16 |
| Hardware | Apple M1, 8 GB unified memory, Metal, 68.25 GB/s |
| Ollama | 0.24.0, default server settings (no flash attention or KV cache env vars) |
| Model | qwen2.5:3b-instruct-q4_K_M, ctx 8192 |
| Estimated decode | 23.0 tok/s (est.), GPU 100% |
| Measured decode | 10.8 tok/s (median of 3 runs via `ollama-compat bench --installed-only`); cross-check with `ollama run --verbose`: 10.87 tok/s |
| Measured prefill | 1365 tok/s; measured GPU fraction 100% |
| Error | estimate about 2.1x too high (measured is 47% of estimate) |

Note: system memory was under pressure during the run (about 24% free). Small
models may also have fixed per-token overhead that a pure bandwidth model ignores.
The constants (`EffGPU 0.65`) are unchanged for v1 because one data point on a
memory-constrained machine is not enough to recalibrate. More data points,
especially from 16 GB and larger Apple Silicon and NVIDIA machines, are needed.
Users can contribute with `ollama-compat bench --json`.

| Field | Value |
|---|---|
| Date | 2026-11-17 |
| Hardware | Apple M1, 8 GB unified memory, Metal, 68.25 GB/s (same machine as above) |
| Ollama | 0.24.0, default server settings |
| Models | qwen2.5:3b-instruct-q4_K_M (1.93 GB), tinyllama:latest (0.64 GB), ctx 8192 |
| Measured decode | qwen2.5 3B: 20.6 tok/s (`bench --installed-only`, median of 3), 21.4 and 21.8 via the API; tinyllama: 64.5 and 65.3 via the API |
| Implied efficiency | tok/s x model bytes / bandwidth: 0.58 to 0.61 (qwen2.5 3B), 0.61 (tinyllama) |
| Estimate before | 23.0 tok/s for qwen2.5 3B (EffGPU 0.65) |

Result: the 10.8 tok/s entry above was memory pressure, not the model. On an
idle machine the bandwidth model holds for both a 0.6 GB and a 1.9 GB model, so
Metal gets its own factor, `EffMetal 0.60` (qwen2.5 3B now estimates 21 tok/s).
Discrete GPUs keep `EffGPU 0.65` until someone measures one.

From 2026-11-17 on, measurements are kept as `bench --share` records in
`calibration/data/` and fitted automatically; see `calibration/README.md`. The
first record (same M1, 20.1 tok/s for qwen2.5 3B) fits a Metal efficiency of
0.568, replacing the hand-set 0.60 above.

## Freshness check

`scripts/check-catalog` compares `sources.yaml` with the Ollama library and
writes a markdown report; it never edits the catalog. It lists tags whose
registry manifest now 404s (a regen would fail), new `q4_K_M`, `q8_0` and
`q3_K_M` tags in tracked families, and library models that are not tracked.

```bash
go run ./scripts/check-catalog -out report.md
```

The weekly `catalog-check` workflow (Mondays 06:00 UTC, or run it manually)
keeps one open issue, "Catalog check: Ollama library changes", labelled
`catalog`, with the latest report, and closes it when nothing is left.
`catalog/check-ignore.txt` lists what was reviewed and not tracked: model
names, or full `family:tag` names for single tags. It starts with a
2026-11-17 baseline of the whole library, minus a short watch list of newer
models worth adding (gemma4, qwen3.5 to 3.8, granite4, llama4 and others),
so the issue shows only those plus anything released later. The tool scrapes
ollama.com HTML, so a page that parses to zero entries fails the run instead
of reporting nothing new.

## Added 2026-11-17

gpt-oss 20B, Qwen3 4B and 30B-A3B 2507 (instruct and thinking), Qwen3 Coder
30B-A3B, Devstral Small 24B, Magistral Small 24B, Phi-4 Reasoning 14B and
Mini Reasoning 3.8B, the DeepSeek-R1 0528 Qwen3 8B distill, Qwen2.5-VL 3B,
7B and 32B, and Qwen3-VL 2B, 4B, 8B, 30B-A3B and 32B. Their quality scores
are maintainer-curated estimates on the same scale. Known approximations:
gpt-oss alternates sliding-window and full attention, so its KV cache is
overstated (conservative); `gpt-oss:20b` has no quant suffix and is marked
`mxfp4` through the `quant` field. Hybrid-attention families are handled since the
Gemma 4 and Qwen3.5 additions below.

## Gemma 4 and Qwen3.5 (2026-11-17)

Gemma 4 (E2B, E4B, 12B, 26B A4B, 31B) and Qwen3.5 (0.8B, 2B, 4B, 9B, 27B,
35B A3B) use hybrid attention, so `gen-catalog` now reads the KV layout from
the GGUF instead of assuming every layer caches every token:

| Field | Meaning | From |
|---|---|---|
| `kv_full_elems` | K+V elements per token in layers whose cache grows with the context | per-layer `head_count_kv`, `key_length`, `value_length`; skips recurrent and shared-KV layers |
| `kv_swa_elems`, `swa_window` | same for sliding-window layers, capped at the window | `sliding_window_pattern`, `key_length_swa`, `sliding_window` |
| `state_bytes` | fixed memory for recurrent (gated delta net) layers | `ssm.*` sizes x recurrent layers x 25 copies |
| `variants[].extra_bytes` | vision projector, loaded with the model but not read per token | the manifest's projector layer |

Models without these GGUF keys keep the plain formula and are unchanged.
Gemma 3 and Qwen3-VL 2B to 30B gained `extra_bytes` (their projector, about
0.85 GB, was not counted before).

Checked against Ollama 0.24.0 on an M1 8 GB with qwen3.5:4b: its KV cache
grew 32 KB per token from 2K to 32K context, matching `kv_full_elems` at f16,
on top of a fixed 1.25 GiB that matches `state_bytes` (Ollama keeps about 25
checkpoint copies of the recurrent state). Ollama also reported 0.3 to 0.7 GB
more compute graph than the 0.5 GB runtime overhead allows for, so hybrid and
vision models can need somewhat more memory than estimated. Gemma 4's
sliding-window layout is computed the same way but not yet measured.

Quality scores for both families are provisional estimates: they are newer
than the benchmark data the maintainer scored the rest of the catalog with.
Active parameters for the E-models are their effective sizes (per-layer
embeddings are looked up, not streamed), and for the MoE models come from
the model names (A4B, A3B).

