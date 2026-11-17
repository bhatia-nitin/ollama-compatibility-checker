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

Automatic: when a `calibration` issue is opened or edited, the
`calibration` workflow parses the pasted JSON with `scripts/add-calibration`
(unknown fields, implausible numbers and markup are rejected), saves it as
`data/YYYY-MM-DD-<hardware>-<ram>gb-issue<N>.json`, refits, runs the tests and
opens a pull request that closes the issue. If the record cannot be read, it
comments on the issue with the reason. Review the PR (watch for speeds far
below the estimate, a sign of memory pressure) and merge it.

Manual: copy the JSON to `data/`, run `go run ./scripts/fit-calibration`,
and commit the record and `internal/calib/factors.json` together.

`go test ./...` (and so CI) fails if `factors.json` does not match a fresh fit
of `data/`, or if any record is malformed.

Skip records taken under memory pressure: on an 8 GB M1 that halved the
measured speed (see the calibration log in `catalog/README.md`).
