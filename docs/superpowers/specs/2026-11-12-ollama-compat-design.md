# Ollama Compatibility Checker: Design

Date: 2026-11-12
Status: Draft, awaiting approval

## Intent

People who want to run open-source LLMs locally with Ollama do not know which model their hardware can run well, or which Ollama settings to use. They guess, pull a 20 GB model, find it runs at 3 tok/s because it spilled out of VRAM, and start over.

This tool scans the machine (GPUs, VRAM, system RAM, CPU), predicts tokens/sec for a catalog of open-source models, and tells the user:

- which model will be **fastest**,
- which model is the **best balance** of quality and speed (the main recommendation),
- the exact **Ollama config** to run it (model tag with quantization, `num_ctx`, `num_gpu`, env vars).

Optionally, and only on explicit request, it downloads a shortlist of models and benchmarks them through the user's local Ollama to replace estimates with measured numbers.

Free and open source. Runs on Windows, macOS, and Linux.

## Goals

1. Single self-contained binary per platform; no runtime to install.
2. Detect GPUs from NVIDIA, AMD, Apple Silicon, and Intel, plus system RAM and CPU, on all three OSes.
3. Default run is instant, offline, and downloads nothing.
4. Produce three picks (Fastest, Balanced, Largest that fits) with estimated tok/s, memory use, and copy-paste Ollama config.
5. Optional benchmark mode that measures real tok/s via the local Ollama API.
6. Machine-readable output (`--json`) so others can build on it.

## Non-goals (v1)

- Runtimes other than Ollama (llama.cpp direct, LM Studio, vLLM).
- GUI or desktop app.
- Fine-grained multi-GPU tensor-split tuning.
- Prompt-processing (prefill) speed as a ranking input. It is reported in benchmark mode only.
- Recommending models by task (coding vs chat). Catalog may carry tags for later.

## Anti-goals (failure even if it technically works)

- **Downloading anything without explicit, informed consent.** The user must see the total download size and confirm before any pull.
- **Any network call in the default run.** No telemetry, ever.
- **Recommending a model that does not fit** and silently runs on CPU.
- **Presenting estimates as measurements.** Estimated numbers are always labeled `est.`; measured numbers are labeled `measured`.
- Leaving gigabytes of benchmark models on disk without telling the user.

## Constraints

- Language: Go. Pure Go, `CGO_ENABLED=0`, so cross-compiling is trivial. No NVML or ROCm library bindings; shell out to vendor CLIs instead.
- Targets: `windows/amd64`, `windows/arm64`, `darwin/arm64`, `darwin/amd64`, `linux/amd64`, `linux/arm64`.
- Must degrade gracefully: if a vendor tool is missing, report what is known and fall back to CPU estimates rather than failing.
- Ollama API default host `http://127.0.0.1:11434`, overridable by `OLLAMA_HOST` (same env var Ollama uses).

## Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **Go CLI, single binary** | Easy distribution, same language as Ollama, trivial cross-compile. GPU detection via vendor CLIs and OS APIs, not native libraries. | **Chosen** by the user. |
| Python CLI (pip/pipx) | Fastest to write, good GPU libs (`pynvml`). Requires Python; Windows users excluded; PyInstaller binaries are fragile and AV-flagged. | Rejected. |
| Rust CLI | Single binary, strong `nvml-wrapper`/`sysinfo`. Slower to build, smaller contributor pool in the Ollama community. | Rejected. |

Data approach: **estimate first, optional benchmark** (user's choice). Estimate-only would be too rough; benchmark-only would force large downloads.

## User experience

Binary name: `ollama-compat` (builder's call to keep or change).

```
ollama-compat                 # = recommend: scan + estimate + picks. Offline.
ollama-compat scan            # hardware report only
ollama-compat recommend       # explicit form of default
    --min-tps 20              # Balanced pick must reach this est. tok/s (default 20)
    --ctx 8192                # context length to size KV cache for (default 8192)
    --top 3                   # how many alternatives to list per pick
ollama-compat bench           # opt-in benchmark of shortlist
    --yes                     # skip confirmation (for scripts)
    --max-download 20GB       # refuse if total pull exceeds this
    --installed-only          # benchmark only models already pulled; zero download
    --keep                    # keep pulled models (default: offer to remove)
ollama-compat catalog         # list models in the embedded catalog
global: --json, --catalog <path>, --no-color
```

Default output sketch:

```
Hardware
  GPU 0  NVIDIA GeForce RTX 4070  12.0 GB VRAM  504 GB/s  CUDA (driver 560.94)
  RAM    32 GB DDR5  ~60 GB/s
  CPU    AMD Ryzen 7 7700X  8C/16T  AVX2 AVX-512

Recommendations  (ctx 8192, estimates)
  Fastest    llama3.2:3b-instruct-q4_K_M    ~142 tok/s est.   2.6 GB  100% GPU
  Balanced   qwen2.5:14b-instruct-q4_K_M    ~ 38 tok/s est.   9.8 GB  100% GPU
  Largest    gemma2:27b-instruct-q3_K_M     ~  9 tok/s est.  14.1 GB   78% GPU

Config for Balanced
  ollama pull qwen2.5:14b-instruct-q4_K_M
  env:   OLLAMA_FLASH_ATTENTION=1  OLLAMA_KV_CACHE_TYPE=q8_0
  Modelfile:
    FROM qwen2.5:14b-instruct-q4_K_M
    PARAMETER num_ctx 8192
    PARAMETER num_gpu 99

Run `ollama-compat bench` to measure these (downloads ~22 GB, asks first).
```

Model names and numbers above are illustrative only.

## Architecture

```
cmd/ollama-compat/        main, cobra-style command wiring
internal/hw/              hardware detection
  gpu_nvidia.go           nvidia-smi --query-gpu=... --format=csv
  gpu_amd_linux.go        rocm-smi --json, fallback /sys/class/drm/*/device/mem_info_vram_total
  gpu_apple.go            sysctl hw.memsize, machdep.cpu.brand_string, system_profiler SPDisplaysDataType -json
  gpu_windows.go          registry HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-...}\*\HardwareInformation.qwMemorySize
  gpu_intel.go            Intel Arc/iGPU via lspci / registry
  mem.go, cpu.go          RAM size, CPU model, cores, AVX2/AVX-512 flags
  bandwidth.go            lookup tables: GPU name -> GB/s, Apple chip -> GB/s, RAM type fallback
internal/catalog/         embedded models.json (go:embed) + schema + loader
internal/estimate/        fit + tok/s model
internal/recommend/       pick selection + config generation
internal/bench/           Ollama API client: list, pull, generate, ps, delete
internal/output/          table and JSON renderers
```

Each detector returns `[]GPU` and a list of warnings; detectors never abort the run. Detection parsers take raw command output as input so they are unit-testable against captured fixtures.

### Hardware model

```go
type GPU struct {
  Vendor        string  // nvidia | amd | apple | intel
  Name          string
  VRAMBytes     uint64  // for Apple: usable unified memory share, see below
  FreeVRAMBytes uint64  // 0 if unknown
  BandwidthGBs  float64 // from table; 0 if unknown
  Backend       string  // cuda | rocm | metal | vulkan | cpu
  Supported     bool    // will Ollama actually use it
}
```

- **Apple Silicon:** unified memory. Usable GPU memory = ~75% of RAM for machines over 36 GB, ~66% at or below (macOS default `iogpu.wired_limit_mb` behavior). Bandwidth from a chip table (M1 through current, base/Pro/Max/Ultra).
- **Windows VRAM:** read `qwMemorySize` from the registry. Do not use WMI `AdapterRAM` (32-bit, caps at 4 GB).
- **Supported flag:** NVIDIA needs compute capability 5.0+; AMD needs a ROCm-supported gfx target (table); Intel and unsupported AMD are marked `Supported=false` and treated as CPU for estimation, with a warning that Ollama's Vulkan backend may change this.
- **Multi-GPU:** if multiple supported GPUs of the same vendor, pool VRAM, use the minimum bandwidth. Mixed vendors: use the largest supported GPU only and warn.

### Catalog

`catalog/models.json`, embedded at build time; `--catalog` overrides. One entry per model, with variants per quantization:

```json
{
  "family": "qwen2.5",
  "name": "Qwen2.5 14B Instruct",
  "params_b": 14.8,
  "active_params_b": 14.8,
  "layers": 48,
  "kv_heads": 8,
  "head_dim": 128,
  "max_ctx": 32768,
  "quality": 72,
  "variants": [
    {"tag": "qwen2.5:14b-instruct-q4_K_M", "size_bytes": 8988110400, "bits": 4.8},
    {"tag": "qwen2.5:14b-instruct-q8_0",   "size_bytes": 15701597504, "bits": 8.5}
  ]
}
```

- `active_params_b` differs from `params_b` for mixture-of-experts models, which run faster than their size suggests.
- `quality` is a 0-100 curated score per model family/size, derived from public benchmark aggregates and documented in `catalog/README.md` with its source. Quantization applies a penalty in code (q8: 0, q6: -1, q5: -2, q4_K_M: -3, q3: -8, q2: -15).
- Initial catalog: about 25-40 current popular open-weight models from the Ollama library, spanning 0.5B to 70B+, including at least one MoE. Exact list is the builder's call, verified against ollama.com tags at build time.
- A `scripts/` helper may verify sizes against the Ollama registry; it is a dev tool, not shipped behavior.

### Estimation

**Memory needed** for a variant at context `C`:

```
kv_bytes   = 2 * layers * kv_heads * head_dim * C * kv_bytes_per_elem   (q8_0 cache: 1, f16: 2)
need_bytes = size_bytes + kv_bytes + 0.5 GB runtime overhead
```

Usable VRAM = total VRAM minus 10% headroom (or free VRAM if known and smaller).

**Fit:** `gpu_fraction = min(1, usable_vram / need_bytes)`, quantized to whole layers.

**Decode tok/s** (memory-bandwidth bound):

```
bytes_per_token = size_bytes * (active_params_b / params_b)
t_gpu = bytes_per_token * gpu_fraction       / (gpu_bw * eff_gpu)
t_cpu = bytes_per_token * (1 - gpu_fraction) / (ram_bw * eff_cpu)
tok_s = 1 / (t_gpu + t_cpu)
```

Starting constants: `eff_gpu = 0.65`, `eff_cpu = 0.55`, per backend overrides allowed (Metal, ROCm). These live in one file so benchmark data can recalibrate them. Target accuracy: within ±30% of measured on full-GPU fits.

Models whose `need_bytes` exceeds VRAM + 80% of system RAM are excluded entirely.

### Picks

- **Fastest:** highest est. tok/s among variants that fit 100% on GPU and have `quality >= 40` (stops a 0.5B toy from always winning). If nothing fits on GPU, highest tok/s overall.
- **Balanced (headline recommendation):** highest quality among variants with est. tok/s >= `--min-tps` (default 20). Ties broken by tok/s.
- **Largest that fits:** highest quality among variants with est. tok/s >= 5 (usable but slow), allowing partial offload.

Each pick lists `--top` runners-up. If no variant meets a threshold, say so plainly and show the closest.

### Config generation

For each pick:

- `num_ctx`: the `--ctx` value, capped at `max_ctx`, and reduced if KV cache would push the model off GPU (report the reduction).
- `num_gpu`: `99` (all layers) if full fit; otherwise the computed layer count.
- Env vars: `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` on CUDA/Metal/ROCm. Omitted on CPU-only.
- Output both a Modelfile snippet and the env vars with per-OS syntax (`export`, `$env:`, `setx`/`launchctl setenv` note).

### Benchmark mode (opt-in)

1. Check Ollama is reachable (`GET /api/version`). If not: clear message with install link, exit non-zero.
2. Shortlist = the three picks (deduplicated). `GET /api/tags` to see what is already installed.
3. Show a table: model, size, already installed yes/no, **total to download**. Then prompt `Download 21.4 GB and benchmark 3 models? [y/N]`. Default No. `--yes` skips; `--max-download` caps; `--installed-only` skips missing models. Non-interactive stdin without `--yes` = refuse.
4. Pull with `POST /api/pull` (streamed), show progress. Ctrl+C cancels cleanly.
5. For each model: one warm-up `POST /api/generate` (load), then 3 runs of a fixed prompt with `options: {num_predict: 256, num_ctx: <ctx>, temperature: 0, seed: 42}`. Decode tok/s = `eval_count / eval_duration`; prefill tok/s = `prompt_eval_count / prompt_eval_duration`. Report the median.
6. `GET /api/ps` after load: report `size_vram / size` as measured GPU fraction, flag if it differs from the estimate.
7. Print estimate vs measured side by side; re-rank picks on measured numbers.
8. Unless `--keep`: list models pulled during this run and ask `Remove them to free 21.4 GB? [Y/n]`. Never remove models the user had before the run.

Benchmark results are printed (and in `--json`) only. Nothing is uploaded.

## Testing

- Detector parsers: table tests against captured fixture outputs (`testdata/nvidia-smi-*.csv`, `rocm-smi-*.json`, `system_profiler-*.json`, registry dumps).
- Estimator and picker: pure functions, table tests with synthetic hardware.
- Bench client: tests against an `httptest` fake Ollama server, including streamed pull and cancellation.
- CI: GitHub Actions matrix on ubuntu, macos, windows runners: `go vet`, `go test`, build all six targets.

## Distribution

- GoReleaser on tag push: GitHub Releases with archives and checksums for all six targets.
- License: MIT (my call; easy to change).
- Later, not v1: Homebrew tap, Scoop/winget, install script.

## Decided by the user

- Estimate first; benchmark is optional and explicit because it downloads a lot of data.
- Go, command-line, single binary.

## Calls made in this design (reviewable)

- Binary name `ollama-compat`, default command = recommend.
- Three picks (Fastest, Balanced, Largest) with `--min-tps 20` and `--ctx 8192` defaults.
- Curated `quality` score in the catalog plus a quantization penalty.
- Bandwidth from lookup tables, not measured.
- MIT license.
- Benchmark offers cleanup of models it pulled.

## Resolved after builder review (my calls, reviewable)

1. **Catalog source.** Generated, not hand-typed. A dev script `scripts/gen-catalog` reads a hand-maintained input list (`catalog/sources.yaml`: tag list + quality score + source per family) and fetches, for each tag, the Ollama registry manifest (`registry.ollama.ai/v2/library/<model>/manifests/<tag>`) for `size_bytes` and reads the GGUF header of the model blob via an HTTP range request for `block_count` (layers), `attention.head_count_kv`, `head_dim` (or `embedding_length / head_count`), `expert_count`/`expert_used_count` (for `active_params_b`). Output `catalog/models.json` is committed and embedded. A unit test validates every catalog entry (non-zero fields, active <= total, tags unique). The script is a build-time dev tool; the shipped binary never calls it.
2. **Accuracy target is an aspiration, not a release gate.** v1 ships with the starting constants and a README caveat. One calibration point is taken on the maintainer's Apple Silicon Mac via bench mode before release. Bench `--json` output includes hardware + estimate + measured so users can file calibration data in GitHub issues (manual, opt-in; no upload from the tool).
3. **Quality scale.** Cross-family comparable 0-100. Builder picks one public aggregate as the single source, normalizes it so the best catalog model is about 90 and a 1B model about 30, and records source + date in `catalog/README.md`. The `quality >= 40` floor and quant penalties are defined on that scale. Models with no score in the chosen source are excluded from picks (still listed by `catalog`).
4. **Intel / Vulkan.** v1 detects and reports Intel GPUs and unsupported AMD GPUs, estimates them as CPU, and prints a note: "Ollama's experimental Vulkan backend may use this GPU; run `bench` to measure." No Vulkan estimation path in v1.
5. **Apple Silicon is special-cased.** Unified memory: there is no separate VRAM/RAM split. GPU memory pool = the 66%/75% share of RAM. A model that does not fit the GPU share is estimated with CPU spill at the **same** bandwidth with `eff_cpu`, and the hard exclusion is `need_bytes > 80% of total RAM` (not VRAM + 80% RAM, which would double-count).
6. **KV cache type.** The estimate assumes `q8_0` KV (1 byte/elem) when the backend is CUDA/Metal/ROCm and the catalog entry has `kv_quant_ok: true` (default true); otherwise f16 (2 bytes) and the env var is not recommended for that pick. Because `OLLAMA_KV_CACHE_TYPE` is server-wide, the output notes that it affects all models.
7. **`num_gpu` layer count (partial fit).** `per_layer = size_bytes / (layers + 1)` (the +1 approximates embeddings/output). `kv_per_layer = kv_bytes / layers`. `num_gpu = floor((usable_vram - 0.5 GB - per_layer) / (per_layer + kv_per_layer)) - 1` (one layer safety margin), clamped to `[0, layers]`. Approximate is acceptable; bench mode reports `/api/ps` actual and flags mismatches.
8. **Remote Ollama.** If `OLLAMA_HOST` is not loopback, `bench` refuses unless `--allow-remote`, and then warns that measurements describe the remote machine, not the scanned one. `--max-download` counts only models not already installed. The shortlist is deduplicated by tag; different quants of one model are separate entries.
9. **Windows detection.** Registry class key gives both `DriverDesc` (name, used for bandwidth lookup and vendor) and `HardwareInformation.qwMemorySize`. NVIDIA on Windows also uses `nvidia-smi` (ships with the driver). No dxdiag/CIM/vulkaninfo in v1. If `compute_cap` is missing from `nvidia-smi`, assume supported and warn.
10. **Unknown GPU bandwidth.** Never silently drop. Fallback heuristic by vendor and VRAM class (for example NVIDIA 8 GB ~ 300 GB/s, 12 GB ~ 450, 16 GB ~ 500, 24 GB ~ 900), marked `low confidence` in output and JSON, with a hint to run `bench` and to open an issue adding the GPU to the table.

## Left to the builder

- Exact catalog contents and quality scores (with documented source).
- CLI library choice (cobra or stdlib `flag`), table rendering library.
- Exact GPU bandwidth table entries.
- Fixed benchmark prompt text.
