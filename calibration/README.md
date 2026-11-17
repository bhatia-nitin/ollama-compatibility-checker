# Calibration data

Each file in `data/` is one `ollama-compat bench --share` record: hardware
model names, memory sizes, the estimate and the measured decode speed for each
benchmarked model. No hostname, username or paths.

`internal/calib/factors.json` is fitted from these files and compiled into the
binary. For each GPU backend (`metal`, `cuda`, `rocm`) the fit uses only runs
that were fully on GPU (estimated and measured) on a GPU with known bandwidth:

- **Efficiency** = measured tok/s x bytes read per token / bandwidth. The
  factor is the median over all such runs. It replaces the default (0.65) for
  that backend.
- **Error** = the largest relative miss of that median on the same runs,
  rounded up to 5 percent, never below 15. It is the `±` shown next to every
  fully-on-GPU speed on that backend.

Backends with no data keep the default factor and `±30%`.

## Adding a record (maintainers)

1. Copy the JSON from the issue to `data/YYYY-MM-DD-<chip>-<memory>.json`,
   for example `data/2026-11-17-apple-m1-8gb.json`.
2. Run `go run ./scripts/fit-calibration` and commit both the record and the
   updated `internal/calib/factors.json`.

`go test ./...` (and so CI) fails if `factors.json` does not match a fresh fit
of `data/`, or if any record is malformed.

Skip records taken under memory pressure: on an 8 GB M1 that halved the
measured speed (see the calibration log in `catalog/README.md`).
