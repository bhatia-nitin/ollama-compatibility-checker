# ollama-compat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a cross-platform Go CLI that detects GPUs/RAM/CPU, estimates Ollama tok/s for a model catalog, recommends Fastest/Balanced/Largest picks with copy-paste Ollama config, and optionally benchmarks a shortlist with explicit download consent.

**Architecture:** Pure-Go binary (`CGO_ENABLED=0`). `internal/hw` parses vendor CLI / OS output into a `Hardware` value; `internal/catalog` embeds a generated `models.json`; `internal/estimate` and `internal/recommend` are pure functions over those; `internal/bench` talks to the Ollama HTTP API; `internal/output` renders tables or JSON; `cmd/ollama-compat` wires cobra commands.

**Tech Stack:** Go 1.24+, `github.com/spf13/cobra`, `golang.org/x/sys/windows/registry`, stdlib `text/tabwriter`, `net/http/httptest` for tests, GoReleaser, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-12-ollama-compat-design.md` (read it; the "Resolved after builder review" section overrides earlier sections where they differ).

## Global Constraints

- Module path: `github.com/shra26/ollama-compatibility-checker`. Binary name: `ollama-compat`.
- `CGO_ENABLED=0`. No NVML/ROCm bindings; shell out to `nvidia-smi`, `rocm-smi`, `system_profiler`, `sysctl`, `lspci`.
- Targets: `windows/amd64`, `windows/arm64`, `darwin/arm64`, `darwin/amd64`, `linux/amd64`, `linux/arm64`.
- Default run (`ollama-compat`, `scan`, `recommend`, `catalog`) makes **zero network calls**. No telemetry anywhere.
- No download without the user seeing total bytes and confirming; prompt default is No; non-interactive stdin without `--yes` refuses.
- Estimated values labeled `est.`; measured labeled `measured`; heuristic bandwidth labeled `low confidence`.
- Ollama host: `OLLAMA_HOST` env, default `http://127.0.0.1:11434`.
- Defaults: `--min-tps 20`, `--ctx 8192`, `--top 3`. Constants: `eff_gpu 0.65`, `eff_cpu 0.55`, VRAM headroom 10%, runtime overhead 0.5 GB, Fastest quality floor 40, Largest tok/s floor 5, exclusion RAM share 80%.
- Quant quality penalty: q8 0, q6 -1, q5 -2, q4 (q4_K_M, q4_0) -3, q3 -8, q2 -15.
- Detectors never abort: they return results plus `[]string` warnings.
- No em dashes in any user-facing text, docs, or commit messages.
- License MIT. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **CPU-only machine (no GPU or all GPUs unsupported):** picks still produced from CPU estimates, scan shows "No supported GPU", no crash. Test in Task 9 and Task 10.
2. **Vendor tool present but failing** (`nvidia-smi` exits non-zero: "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"): detector returns zero GPUs plus a warning, not an error. Test in Task 3.
3. **Nothing fits** (4 GB RAM laptop): each pick says "no model meets threshold" and shows the closest; exit code 0. Test in Task 10.
4. **`--ctx` above a model's `max_ctx`, or KV pushes model off GPU:** ctx capped and the reduction reported in the config. Test in Task 10.
5. **Ctrl+C during `bench` pull, or stdin not a TTY:** pull context cancelled, no cleanup prompt for models never completed; non-TTY without `--yes` exits non-zero before any network call except `/api/version` and `/api/tags`. Test in Task 13.

---

### Task 1: Scaffold module, CLI root, CI, license

**Files:**
- Create: `go.mod`, `cmd/ollama-compat/main.go`, `cmd/ollama-compat/root.go`, `cmd/ollama-compat/root_test.go`, `LICENSE`, `.github/workflows/ci.yml`
- Modify: `.gitignore` (add `dist/`, `/ollama-compat`, `*.exe`)

**Interfaces:**
- Produces: `func newRootCmd(stdout, stderr io.Writer) *cobra.Command` with persistent flags `--json` (bool), `--catalog` (string), `--no-color` (bool); `var version = "dev"` (set by ldflags `-X main.version`).

- [ ] **Step 1:** Prerequisite: `go version` must print 1.24 or newer. If missing on macOS: `brew install go`.
- [ ] **Step 2: Write failing test** `TestVersionFlag`: execute root with `--version`, assert stdout contains `ollama-compat dev`.
- [ ] **Step 3:** Run `go test ./cmd/...` Expected: FAIL (package does not compile).
- [ ] **Step 4:** `go mod init github.com/shra26/ollama-compatibility-checker`; implement `newRootCmd` with cobra, `Use: "ollama-compat"`, `Version: version`. `main` calls `newRootCmd(os.Stdout, os.Stderr).Execute()` and exits 1 on error.
- [ ] **Step 5:** Run `go test ./... && go vet ./...` Expected: PASS.
- [ ] **Step 6:** Add MIT `LICENSE` (copyright "2026 ollama-compatibility-checker contributors") and `ci.yml`: matrix `ubuntu-latest, macos-latest, windows-latest`, steps `actions/setup-go` (go-version-file: go.mod), `go vet ./...`, `go test ./...`; plus one ubuntu job cross-building all six targets with `CGO_ENABLED=0 GOOS=.. GOARCH=.. go build ./cmd/ollama-compat`.
- [ ] **Step 7: Commit** `feat: scaffold ollama-compat CLI`

### Task 2: Hardware types and bandwidth tables

**Files:**
- Create: `internal/hw/types.go`, `internal/hw/bandwidth.go`, `internal/hw/bandwidth_test.go`

**Interfaces:**
- Produces:
  ```go
  type GPU struct {
      Vendor        string  // "nvidia" | "amd" | "apple" | "intel"
      Name          string
      VRAMBytes     uint64
      FreeVRAMBytes uint64
      BandwidthGBs  float64
      BandwidthLowConfidence bool
      Backend       string  // "cuda" | "rocm" | "metal" | "vulkan" | "cpu"
      Supported     bool
      Driver        string
      ComputeCap    string  // NVIDIA "8.9"; AMD gfx target "gfx1100"
  }
  type CPU struct { Model string; Cores, Threads int; AVX2, AVX512 bool }
  type Hardware struct {
      OS, Arch     string
      GPUs         []GPU
      RAMBytes     uint64
      RAMBandwidthGBs float64
      CPU          CPU
      UnifiedMemory bool // Apple Silicon
      Warnings     []string
  }
  func LookupGPUBandwidth(vendor, name string, vramBytes uint64) (gbs float64, lowConfidence bool)
  func LookupAppleBandwidth(chip string) (gbs float64, ok bool)
  func RAMBandwidthFallback(os, arch string) float64
  ```

- [ ] **Step 1: Write failing tests** (table): `LookupGPUBandwidth("nvidia","NVIDIA GeForce RTX 4090",24<<30)` == (1008, false); `("nvidia","NVIDIA GeForce RTX 3060",12<<30)` == (360, false); unknown `("nvidia","NVIDIA Future 9999",16<<30)` == (500, true); unknown 8 GB -> 300, 12 GB -> 450, 24 GB -> 900 (all true). `LookupAppleBandwidth("Apple M1")` == 68.25; `"Apple M2 Max"` == 400; `"Apple M3 Pro"` == 150; `"Apple M4 Max"` == 546. Name matching is case-insensitive substring on the normalized name with longest key winning ("4070 Ti SUPER" beats "4070").
- [ ] **Step 2:** Run `go test ./internal/hw/` Expected: FAIL.
- [ ] **Step 3:** Implement tables covering NVIDIA RTX 20/30/40/50 series and datacenter A100/H100/L4/T4, AMD RX 6000/7000/9000 and MI series, Intel Arc A/B, Apple M1 through M4 (base/Pro/Max/Ultra). Unknown fallback by VRAM class: <=8 GB 300, <=12 GB 450, <=16 GB 500, <=24 GB 900, larger 1500 (AMD and Intel use the same classes times 0.8). `RAMBandwidthFallback`: darwin/arm64 is not used (Apple table instead); default 50 GB/s.
- [ ] **Step 4:** Run tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(hw): add hardware types and bandwidth tables`

### Task 3: NVIDIA detector

**Files:**
- Create: `internal/hw/gpu_nvidia.go`, `internal/hw/gpu_nvidia_test.go`, `internal/hw/testdata/nvidia-smi-single.csv`, `nvidia-smi-dual.csv`, `nvidia-smi-nocap.csv`
- Create: `internal/hw/exec.go`

**Interfaces:**
- Produces: `type Runner func(ctx context.Context, name string, args ...string) ([]byte, error)`; `var DefaultRunner Runner` (wraps `exec.CommandContext`, 5 s timeout); `func ParseNvidiaSMI(csv []byte) ([]GPU, []string)`; `func DetectNvidia(ctx context.Context, run Runner) ([]GPU, []string)`.
- Command: `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader,nounits` (memory in MiB).

- [ ] **Step 1: Write failing tests:** `TestParseNvidiaSMISingle` (fixture `NVIDIA GeForce RTX 4070, 12282, 11500, 560.94, 8.9`) asserts 1 GPU, VRAM 12282 MiB in bytes, Backend "cuda", Supported true, Bandwidth 504. `TestParseNvidiaSMIOldCard` (`compute_cap 3.5`) asserts Supported false. `TestParseNvidiaSMINoCap` (fixture with `[N/A]` for compute_cap) asserts Supported true and one warning containing "compute capability". `TestDetectNvidiaToolFails`: runner returns error "exit status 9" and output "NVIDIA-SMI has failed"; asserts 0 GPUs, 1 warning, no panic. `TestDetectNvidiaNotInstalled`: runner returns `exec.ErrNotFound`; asserts 0 GPUs, 0 warnings.
- [ ] **Step 2:** Run `go test ./internal/hw/ -run Nvidia` Expected: FAIL.
- [ ] **Step 3:** Implement. Supported = compute cap >= 5.0.
- [ ] **Step 4:** Run tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(hw): detect NVIDIA GPUs via nvidia-smi`

### Task 4: AMD Linux detector

**Files:**
- Create: `internal/hw/gpu_amd.go`, `internal/hw/gpu_amd_test.go`, `internal/hw/testdata/rocm-smi.json`, `internal/hw/testdata/sysfs/` (fake `/sys/class/drm/card0/device/{vendor,mem_info_vram_total,product_name}`)

**Interfaces:**
- Produces: `func ParseRocmSMI(js []byte) ([]GPU, []string)`; `func DetectAMDSysfs(root string) ([]GPU, []string)` (root = "/" in prod, testdata dir in tests); `func DetectAMD(ctx context.Context, run Runner, sysRoot string) ([]GPU, []string)`; `var rocmSupportedTargets map[string]bool`.
- Command: `rocm-smi --showproductname --showmeminfo vram --showhw --json`.

- [ ] **Step 1: Write failing tests:** `TestParseRocmSMI` (RX 7900 XTX fixture, gfx1100) asserts VRAM 24 GiB, Backend "rocm", Supported true. `TestAMDUnsupportedTarget` (gfx1035 iGPU) asserts Supported false, Backend "cpu". `TestDetectAMDSysfsFallback`: runner returns `exec.ErrNotFound`, sysfs fixture with vendor `0x1002` and vram 17163091968 asserts one GPU with that VRAM and a warning "rocm-smi not found".
- [ ] **Step 2:** Run Expected: FAIL.
- [ ] **Step 3:** Implement. `rocmSupportedTargets`: gfx900, gfx906, gfx908, gfx90a, gfx940, gfx941, gfx942, gfx1030, gfx1100, gfx1101, gfx1102, gfx1200, gfx1201. Sysfs path has no gfx target: mark Supported true only if name matches a bandwidth table entry for a discrete RX/MI card, else false with warning.
- [ ] **Step 4:** Run Expected: PASS.
- [ ] **Step 5: Commit** `feat(hw): detect AMD GPUs on Linux`

### Task 5: Apple Silicon detector

**Files:**
- Create: `internal/hw/gpu_apple.go`, `internal/hw/gpu_apple_test.go`, `internal/hw/testdata/system_profiler-m2pro.json`

**Interfaces:**
- Produces: `func AppleGPUShare(ramBytes uint64) uint64`; `func ParseAppleDisplays(js []byte, chip string, ramBytes uint64) ([]GPU, []string)`; `func DetectApple(ctx context.Context, run Runner) ([]GPU, uint64 /*ram*/, string /*chip*/, []string)`.
- Commands: `sysctl -n hw.memsize`, `sysctl -n machdep.cpu.brand_string`, `system_profiler SPDisplaysDataType -json`.

- [ ] **Step 1: Write failing tests:** `TestAppleGPUShare`: 16 GiB -> 66% (10.56 GiB, compare with integer math `ram*66/100`); 36 GiB -> 66%; 64 GiB -> 75%. `TestParseAppleDisplaysM2Pro`: chip "Apple M2 Pro", 32 GiB RAM asserts Vendor "apple", Backend "metal", Supported true, Bandwidth 200, VRAM = AppleGPUShare(32 GiB). `TestParseAppleDisplaysIntelMac`: discrete AMD Radeon Pro entry asserts Supported false (Ollama does not use GPU on Intel Macs) with warning.
- [ ] **Step 2:** Run Expected: FAIL.
- [ ] **Step 3:** Implement. Threshold: >36 GiB gets 75%, else 66%.
- [ ] **Step 4:** Run Expected: PASS.
- [ ] **Step 5: Commit** `feat(hw): detect Apple Silicon unified memory GPU`

### Task 6: Windows registry detector

**Files:**
- Create: `internal/hw/gpu_windows.go` (`//go:build windows`), `internal/hw/gpu_windows_parse.go` (no build tag), `internal/hw/gpu_windows_parse_test.go`

**Interfaces:**
- Produces: `type RegAdapter struct { DriverDesc string; QwMemorySize uint64; DriverVersion string }`; `func ParseWindowsAdapters(adapters []RegAdapter) ([]GPU, []string)` (portable, tested everywhere); `func DetectWindowsRegistry() ([]RegAdapter, error)` (windows only; reads `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000..00NN`, values `DriverDesc`, `HardwareInformation.qwMemorySize`, `DriverVersion`).

- [ ] **Step 1: Write failing tests:** adapters `[{"AMD Radeon RX 7800 XT", 16 GiB}, {"Microsoft Basic Display Adapter", 0}, {"Intel(R) UHD Graphics 770", 128 MiB}]` asserts AMD GPU with 16 GiB, Backend "rocm", Supported true (Ollama on Windows supports RX 7000); basic display skipped; Intel Supported false, Backend "cpu", warning mentions Vulkan. NVIDIA entries are skipped here (NVIDIA comes from nvidia-smi, Task 3).
- [ ] **Step 2:** Run Expected: FAIL.
- [ ] **Step 3:** Implement both files. Windows AMD supported if name matches RX 6800 and up, RX 7000, RX 9000 series.
- [ ] **Step 4:** Run `go test ./internal/hw/` and `GOOS=windows go vet ./internal/hw/` Expected: PASS.
- [ ] **Step 5: Commit** `feat(hw): detect Windows GPUs from registry`

### Task 7: RAM, CPU, Intel, and Detect() aggregator

**Files:**
- Create: `internal/hw/cpu.go`, `internal/hw/mem.go`, `internal/hw/gpu_intel.go`, `internal/hw/detect.go`, `internal/hw/detect_test.go`, OS-specific `mem_linux.go`, `mem_darwin.go`, `mem_windows.go`

**Interfaces:**
- Consumes: Tasks 2-6 detectors.
- Produces: `type Probes struct { Run Runner; SysRoot string; GOOS, GOARCH string }`; `func Detect(ctx context.Context, p Probes) Hardware`; `func MergeGPUs(gpus []GPU) (pooled []GPU, warnings []string)`.

- [ ] **Step 1: Write failing tests** for `MergeGPUs`: two supported NVIDIA (12 GiB @ 504, 8 GiB @ 272) -> one pooled entry 20 GiB @ 272 named "2x GPU (NVIDIA ...)"; NVIDIA + AMD both supported -> keep larger VRAM one, warning "mixed vendors"; unsupported GPUs kept in list but never pooled. `TestDetectNoGPU` with a runner that returns `exec.ErrNotFound` for everything on GOOS "linux" asserts `len(GPUs)==0`, `RAMBytes>0` is not required (fake), no panic.
- [ ] **Step 2:** Run Expected: FAIL.
- [ ] **Step 3:** Implement. Linux RAM from `/proc/meminfo` (under SysRoot), CPU from `/proc/cpuinfo` flags (`avx2`, `avx512f`); darwin via `sysctl hw.memsize`, `sysctl -n machdep.cpu.brand_string`, `hw.physicalcpu`, `hw.logicalcpu`; windows via `golang.org/x/sys/windows` `GlobalMemoryStatusEx` and `runtime.NumCPU()`, CPU name from registry `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString`. Intel on Linux from `lspci -mm` lines with vendor "Intel" and class "VGA": Supported false. Apple: set `UnifiedMemory=true` and `RAMBandwidthGBs` = GPU bandwidth.
- [ ] **Step 4:** Run `go test ./internal/hw/` plus `GOOS=windows go vet ./...` and `GOOS=darwin go vet ./...` Expected: PASS.
- [ ] **Step 5: Commit** `feat(hw): aggregate hardware detection across OSes`

### Task 8: Catalog schema, loader, generator

**Files:**
- Create: `internal/catalog/catalog.go`, `internal/catalog/catalog_test.go`, `internal/catalog/models.json`, `catalog/sources.yaml`, `catalog/README.md`, `scripts/gen-catalog/main.go`, `scripts/gen-catalog/gguf.go`, `scripts/gen-catalog/gguf_test.go`, `scripts/gen-catalog/testdata/header.gguf`

**Interfaces:**
- Produces:
  ```go
  type Variant struct { Tag string `json:"tag"`; SizeBytes uint64 `json:"size_bytes"`; Bits float64 `json:"bits"`; Quant string `json:"quant"` }
  type Model struct {
      Family, Name string; ParamsB, ActiveParamsB float64
      Layers, KVHeads, HeadDim, MaxCtx int
      Quality int; KVQuantOK bool `json:"kv_quant_ok"`; Variants []Variant
  }
  func Load(r io.Reader) ([]Model, error)
  func Embedded() []Model
  func Validate(ms []Model) error
  func QuantPenalty(quant string) int
  ```
- `KVQuantOK` defaults true when absent (use `*bool` in JSON decode).

- [ ] **Step 1: Write failing tests:** `TestEmbeddedValid` asserts `Validate(Embedded())==nil` and `len>=25` and at least one model with `ActiveParamsB < ParamsB`. `TestValidateRejects`: zero layers, active > total, duplicate tag each return error naming the tag. `TestQuantPenalty`: q8_0 0, q6_K -1, q5_K_M -2, q4_K_M -3, q4_0 -3, q3_K_M -8, q2_K -15, unknown 0. `TestParseGGUFHeader` (gen-catalog): small fixture asserts `block_count`, `head_count_kv`, `key_length`, `expert_count` read correctly.
- [ ] **Step 2:** Run `go test ./internal/catalog/ ./scripts/...` Expected: FAIL.
- [ ] **Step 3:** Implement loader and `go:embed models.json`. Implement `gen-catalog`: read `sources.yaml` (list of `{family, name, params_b, quality, quality_source, tags: [...]}`), for each tag GET `https://registry.ollama.ai/v2/library/<model>/manifests/<tag>` with `Accept: application/vnd.docker.distribution.manifest.v2+json`, take layer with mediaType `application/vnd.ollama.image.model` for size and digest, GET blob `https://registry.ollama.ai/v2/library/<model>/blobs/<digest>` with `Range: bytes=0-8388607`, parse GGUF v3 KV metadata (`<arch>.block_count`, `<arch>.attention.head_count_kv`, `<arch>.attention.key_length` else `embedding_length/head_count`, `<arch>.context_length`, `<arch>.expert_count`, `<arch>.expert_used_count`). Active params for MoE: `params_b * expert_used_count / expert_count` (approximation, note in README). Write sorted JSON.
- [ ] **Step 4:** Fill `sources.yaml` with 25-40 current Ollama library models from 0.5B to 70B+, at least one MoE, two to three quants each (q4_K_M always; q8_0 for <=14B; q3_K_M for >=27B). Pick one public aggregate leaderboard as the quality source; normalize best model to about 90 and a 1B model to about 30; record source name and date in `catalog/README.md`. Run `go run ./scripts/gen-catalog -in catalog/sources.yaml -out internal/catalog/models.json` (network, dev only).
- [ ] **Step 5:** Run tests Expected: PASS.
- [ ] **Step 6: Commit** `feat(catalog): add model catalog, loader and generator`

### Task 9: Estimator

**Files:**
- Create: `internal/estimate/estimate.go`, `internal/estimate/estimate_test.go`

**Interfaces:**
- Consumes: `hw.Hardware`, `catalog.Model`, `catalog.Variant`.
- Produces:
  ```go
  type Params struct { Ctx int; EffGPU, EffCPU float64 } // DefaultParams(): 8192, 0.65, 0.55
  type Estimate struct {
      Model catalog.Model; Variant catalog.Variant
      Ctx int; CtxReduced bool
      KVQuant bool          // q8_0 cache assumed
      NeedBytes uint64; GPUFraction float64; NumGPU int
      TokS float64; Excluded bool; LowConfidence bool
  }
  func KVBytes(m catalog.Model, ctx int, q8 bool) uint64
  func UsableVRAM(h hw.Hardware) uint64
  func NumGPULayers(m catalog.Model, v catalog.Variant, usable, kv uint64) int
  func EstimateVariant(h hw.Hardware, m catalog.Model, v catalog.Variant, p Params) Estimate
  func EstimateAll(h hw.Hardware, ms []catalog.Model, p Params) []Estimate
  ```

- [ ] **Step 1: Write failing tests** with synthetic hardware:
  - `TestKVBytes`: layers 48, kv_heads 8, head_dim 128, ctx 8192, q8 -> `2*48*8*128*8192*1` = 805306368.
  - `TestFullFitRTX4070`: 12 GiB @ 504, model 8.99e9 bytes, active==total, ctx 8192 q8 -> GPUFraction 1, NumGPU 99, TokS within 1% of `0.65*504e9/8.99e9` (about 36.4).
  - `TestPartialOffload`: 8 GiB GPU @ 300, RAM 32 GiB @ 50, 14 GB model -> 0 < GPUFraction < 1, TokS strictly between pure-CPU and pure-GPU values, NumGPU from `NumGPULayers` formula in spec item 7.
  - `TestMoEFaster`: same size_bytes, ActiveParamsB = ParamsB/8 -> TokS 8x the dense estimate.
  - `TestAppleUnified`: UnifiedMemory, 32 GiB RAM, Apple share 66%, 400 GB/s; model needing 24 GiB -> not excluded, partial fraction, CPU term uses the same 400 GB/s with EffCPU; model needing 27 GiB (>80% of 32 GiB) -> Excluded.
  - `TestExcludedNonApple`: need > VRAM + 80% RAM -> Excluded.
  - `TestCPUOnly`: no GPUs -> GPUFraction 0, NumGPU 0, KVQuant false, TokS = `0.55*ramBW/size`.
  - `TestKVQuantNotOK`: model with KVQuantOK false on CUDA -> KVQuant false, KV uses 2 bytes.
  - `TestLowConfidencePropagates`: GPU with BandwidthLowConfidence -> Estimate.LowConfidence true.
- [ ] **Step 2:** Run `go test ./internal/estimate/` Expected: FAIL.
- [ ] **Step 3:** Implement per spec "Estimation" and resolved items 5, 6, 7. GPU pool = first supported (merged) GPU; `UsableVRAM` = min(VRAM*0.9, FreeVRAM if non-zero). `GPUFraction` = `NumGPU/Layers` when partial (whole layers). Ctx = min(p.Ctx, m.MaxCtx); set CtxReduced when capped (KV shrink to keep full fit happens in Task 10).
- [ ] **Step 4:** Run Expected: PASS.
- [ ] **Step 5: Commit** `feat(estimate): add fit and tok/s estimator`

### Task 10: Picks and config generation

**Files:**
- Create: `internal/recommend/recommend.go`, `internal/recommend/config.go`, `internal/recommend/recommend_test.go`, `internal/recommend/config_test.go`

**Interfaces:**
- Consumes: `estimate.EstimateAll`, `estimate.EstimateVariant`, `catalog.QuantPenalty`.
- Produces:
  ```go
  type Options struct { MinTPS float64; Ctx, Top int }
  type Pick struct {
      Kind string // "fastest" | "balanced" | "largest"
      Best *estimate.Estimate; RunnersUp []estimate.Estimate
      Met bool; Closest *estimate.Estimate; Note string
  }
  type Config struct {
      PullCmd string; Modelfile string; NumCtx, NumGPU int; CtxNote string
      Env map[string]string // OLLAMA_FLASH_ATTENTION, OLLAMA_KV_CACHE_TYPE
      EnvSyntax map[string]string // "posix", "powershell", "windows-persistent", "macos-app"
  }
  func Score(e estimate.Estimate) int // Quality + QuantPenalty
  func Recommend(h hw.Hardware, ms []catalog.Model, o Options) []Pick // always 3, in order fastest, balanced, largest
  func BuildConfig(h hw.Hardware, e estimate.Estimate, o Options) Config
  ```

- [ ] **Step 1: Write failing tests:**
  - `TestFastestSkipsToyModels`: 0.5B model quality 25 is fastest but Pick fastest returns the 3B (quality 45).
  - `TestBalancedPicksHighestQualityAboveMinTPS`, `TestLargestAllowsPartialAboveFive`.
  - `TestNothingMeetsThreshold` (4 GiB RAM, no GPU): all three Picks have `Met==false`, non-nil `Closest`, Note non-empty; no panic.
  - `TestRunnersUpCount`: Top 3 gives at most 3 runners-up, sorted by the pick's criterion.
  - `TestConfigFullFit`: CUDA full fit -> NumGPU 99, Env has `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`, Modelfile equals:
    ```
    FROM <tag>
    PARAMETER num_ctx 8192
    PARAMETER num_gpu 99
    ```
  - `TestConfigCPUOnlyNoEnv`: Env empty.
  - `TestConfigCtxCapped`: `--ctx 65536` on model MaxCtx 32768 -> NumCtx 32768, CtxNote mentions 32768.
  - `TestConfigCtxShrinksToFit`: model fits fully at ctx 4096 but not 8192 -> NumCtx is the largest power of two in [2048, ctx] that keeps full fit, CtxNote explains; if none fits, keep requested ctx with partial NumGPU.
  - `TestEnvSyntax`: posix `export OLLAMA_FLASH_ATTENTION=1`, powershell `$env:OLLAMA_FLASH_ATTENTION="1"`, windows-persistent `setx OLLAMA_FLASH_ATTENTION 1`, macos-app `launchctl setenv OLLAMA_FLASH_ATTENTION 1`; KV note text contains "applies to all models".
- [ ] **Step 2:** Run Expected: FAIL.
- [ ] **Step 3:** Implement per spec "Picks" and "Config generation". Ties broken by TokS.
- [ ] **Step 4:** Run Expected: PASS.
- [ ] **Step 5: Commit** `feat(recommend): add picks and Ollama config generation`

### Task 11: Output renderers and offline commands

**Files:**
- Create: `internal/output/table.go`, `internal/output/json.go`, `internal/output/output_test.go`, `cmd/ollama-compat/scan.go`, `cmd/ollama-compat/recommend.go`, `cmd/ollama-compat/catalog.go`, `cmd/ollama-compat/commands_test.go`

**Interfaces:**
- Consumes: `hw.Detect`, `catalog.Embedded`/`Load`, `recommend.Recommend`, `recommend.BuildConfig`.
- Produces: `func RenderScan(w io.Writer, h hw.Hardware, color bool)`; `func RenderRecommend(w io.Writer, h hw.Hardware, picks []recommend.Pick, cfg recommend.Config, o recommend.Options, color bool)`; `func RenderCatalog(w io.Writer, ms []catalog.Model)`; `func WriteJSON(w io.Writer, v any) error`; JSON top-level `{"schema_version": 1, "hardware": ..., "picks": ..., "config": ...}`. Root command with no subcommand runs recommend. Commands take a `detect func(context.Context) hw.Hardware` injected for tests.

- [ ] **Step 1: Write failing tests:** golden-file tests (`testdata/*.golden`, update with `-update` flag) for scan, recommend, CPU-only recommend, nothing-meets-threshold recommend. Assertions inside goldens: every tok/s value is suffixed `est.`; low-confidence GPU shows `low confidence`; footer line `Run \`ollama-compat bench\` to measure these (downloads ~X GB, asks first).` with X computed from picks' sizes; `--json` output parses and has `schema_version` 1. `TestNoNetworkInDefaultRun`: set `http.DefaultTransport` to a transport that fails the test on any request, run root with no args, assert no request made.
- [ ] **Step 2:** Run `go test ./internal/output/ ./cmd/...` Expected: FAIL.
- [ ] **Step 3:** Implement with `text/tabwriter`; color via ANSI only when stdout is a terminal (`golang.org/x/term`) and `--no-color`/`NO_COLOR` not set. Recommend shows config for the Balanced pick (or Closest if not met), env syntax for the current OS only.
- [ ] **Step 4:** Run tests, then `go run ./cmd/ollama-compat` on this machine. Expected: PASS and a sensible Apple Silicon report.
- [ ] **Step 5: Commit** `feat: add scan, recommend and catalog commands`

### Task 12: Ollama API client

**Files:**
- Create: `internal/bench/client.go`, `internal/bench/client_test.go`

**Interfaces:**
- Produces:
  ```go
  type Client struct { BaseURL string; HTTP *http.Client }
  func HostFromEnv() (baseURL string, loopback bool) // OLLAMA_HOST parsing: "0.0.0.0:11434", "host", "http://h:p"
  func (c *Client) Version(ctx context.Context) (string, error)
  func (c *Client) Tags(ctx context.Context) ([]string, error)
  func (c *Client) Pull(ctx context.Context, tag string, progress func(completed, total int64)) error
  type GenResult struct { EvalCount int; EvalDuration time.Duration; PromptEvalCount int; PromptEvalDuration time.Duration }
  func (c *Client) Generate(ctx context.Context, tag, prompt string, numCtx, numPredict int) (GenResult, error) // stream:false, options temperature 0, seed 42
  type PSModel struct { Name string; Size, SizeVRAM int64 }
  func (c *Client) PS(ctx context.Context) ([]PSModel, error)
  func (c *Client) Delete(ctx context.Context, tag string) error
  ```

- [ ] **Step 1: Write failing tests** against `httptest.Server`: `TestHostFromEnv` table (unset -> `http://127.0.0.1:11434`, true; `0.0.0.0` -> `http://127.0.0.1:11434`, true; `gpu-box:11434` -> `http://gpu-box:11434`, false; `http://[::1]:11434` -> loopback true). `TestPullStreamsProgress`: server streams 3 NDJSON lines then `{"status":"success"}`, progress called with increasing values. `TestPullErrorLine`: line `{"error":"pull model manifest: file does not exist"}` -> error containing that text. `TestPullCancel`: cancel ctx mid-stream -> returns `context.Canceled` promptly. `TestGenerateParsesDurations`: eval_duration nanoseconds -> correct Duration. `TestDeleteSendsBody`: DELETE `/api/delete` with `{"model":tag}`.
- [ ] **Step 2:** Run `go test ./internal/bench/` Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): add Ollama API client`

### Task 13: bench command flow

**Files:**
- Create: `internal/bench/run.go`, `internal/bench/run_test.go`, `cmd/ollama-compat/bench.go`

**Interfaces:**
- Consumes: Task 12 client, `recommend.Recommend`, `estimate.Estimate`.
- Produces:
  ```go
  type Options struct {
      Yes, InstalledOnly, Keep, AllowRemote bool
      MaxDownload uint64 // 0 = no cap; parsed from "20GB" (1e9 units, also accepts GiB)
      Ctx, Runs int // Runs default 3
      In io.Reader; Out io.Writer; IsTTY bool
  }
  type Result struct { Tag string; Est estimate.Estimate; DecodeTokS, PrefillTokS float64; MeasuredGPUFraction float64; Mismatch bool; Pulled bool; Err string }
  func Run(ctx context.Context, c *Client, shortlist []estimate.Estimate, o Options) ([]Result, error)
  func ParseSize(s string) (uint64, error)
  const BenchPrompt = "..." // fixed text, about 60 tokens, asks for a short explanation of a technical topic
  ```

- [ ] **Step 1: Write failing tests** with a fake Ollama server recording every request:
  - `TestRefusesNonTTYWithoutYes`: IsTTY false, Yes false, 1 missing model -> error "refusing to download without --yes", recorded requests only `/api/version` and `/api/tags`.
  - `TestPromptDefaultNo`: input `"\n"` -> no `/api/pull`.
  - `TestPromptShowsTotal`: output contains total size of missing models only (installed one excluded) and `[y/N]`.
  - `TestMaxDownloadCap`: total 21 GB, cap 20GB -> error naming both numbers, no pull.
  - `TestInstalledOnly`: zero pulls; missing models listed as skipped.
  - `TestRemoteRefused`: loopback false, AllowRemote false -> error mentioning `--allow-remote`; with AllowRemote -> warning line "describe the remote machine".
  - `TestMedianOfRuns`: 3 runs with eval rates 30, 50, 40 -> DecodeTokS 40; one warm-up request precedes them.
  - `TestMismatchFlag`: est. fraction 1.0, ps reports size_vram/size 0.6 -> Mismatch true.
  - `TestCleanupOnlyPulled`: pre-installed model never deleted; pulled model deleted after `"\n"` (default Yes); with Keep no delete.
  - `TestCancelDuringPull`: ctx cancelled during pull -> no Generate, no cleanup prompt for the incomplete model, returns `context.Canceled`.
  - `TestParseSize`: "20GB" 20e9, "1.5GiB" 1610612736, "abc" error.
- [ ] **Step 2:** Run Expected: FAIL.
- [ ] **Step 3:** Implement per spec "Benchmark mode" and resolved item 8. Shortlist = Best of each Pick, deduplicated by tag. `cmd/bench.go` wires flags `--yes --max-download --installed-only --keep --allow-remote --runs`, uses `signal.NotifyContext(os.Interrupt)`, detects TTY with `term.IsTerminal(int(os.Stdin.Fd()))`, prints estimate vs measured table, re-ranks picks on measured tok/s, and supports `--json` (hardware + estimate + measured).
- [ ] **Step 4:** Run tests. Then manual check on this Mac: `go run ./cmd/ollama-compat bench --installed-only`. Expected: tests PASS; manual run measures any already-installed shortlist model and downloads nothing.
- [ ] **Step 5: Commit** `feat: add opt-in bench command`

### Task 14: Release pipeline and README

**Files:**
- Create: `.goreleaser.yaml`, `.github/workflows/release.yml`
- Modify: `README.md`

- [ ] **Step 1:** `.goreleaser.yaml`: one build `ollama-compat` from `./cmd/ollama-compat`, `CGO_ENABLED=0`, the six targets, ldflags `-s -w -X main.version={{.Version}}`, archives tar.gz (zip on windows), checksums. `release.yml` on tag `v*` runs `goreleaser/goreleaser-action` with `GITHUB_TOKEN`.
- [ ] **Step 2:** Run `goreleaser check` and `goreleaser build --snapshot --clean` (install with `brew install goreleaser` if absent). Expected: config valid, six binaries in `dist/`.
- [ ] **Step 3:** README: what it does, install (download from Releases), usage for each command with the output sketch from the spec, "How estimates work" (bandwidth-bound formula, accuracy caveat: target within plus or minus 30 percent, not guaranteed), privacy ("default run makes no network calls; bench only talks to your local Ollama and asks before downloading"), supported hardware table (NVIDIA CUDA, AMD ROCm, Apple Metal; Intel and unsupported AMD estimated as CPU), contributing calibration data via `bench --json` in an issue, regenerate catalog instructions.
- [ ] **Step 4:** Run `go test ./... && go vet ./...` Expected: PASS.
- [ ] **Step 5: Commit** `docs: add README and release pipeline`

### Task 15: Calibration point on maintainer Mac

**Files:**
- Modify: `internal/estimate/estimate.go` (constants only, if needed), `catalog/README.md` (record result)

- [ ] **Step 1:** With user approval of the download size shown by the prompt, run `go run ./cmd/ollama-compat bench --json > calib.json` on the maintainer's Apple Silicon Mac (or `--installed-only` if they decline downloads).
- [ ] **Step 2:** If the Metal median error across measured models exceeds 30%, set a Metal-specific `EffGPU` override so the median error is under 15% for this machine; update tests that pin `0.65` only for non-Metal backends.
- [ ] **Step 3:** Run `go test ./...` Expected: PASS. Record hardware, models, est. vs measured in `catalog/README.md` "Calibration log". Do not commit `calib.json`.
- [ ] **Step 4: Commit** `chore: calibrate Metal efficiency from first benchmark`
