package hw

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

const mib = uint64(1) << 20

// minComputeCap is the lowest CUDA compute capability Ollama supports.
const minComputeCap = 5.0

// ParseNvidiaSMI parses the output of
// nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader,nounits
// Memory values are MiB. Malformed lines are skipped with a warning.
func ParseNvidiaSMI(csv []byte) ([]GPU, []string) {
	return parseNvidiaSMI(csv, true)
}

// parseNvidiaSMI parses the 5-field form, or with withCap false the 4-field
// form without compute_cap, where the capability is treated as unknown.
func parseNvidiaSMI(csv []byte, withCap bool) ([]GPU, []string) {
	nf := 4
	if withCap {
		nf = 5
	}
	var gpus []GPU
	var warns []string
	for _, line := range strings.Split(string(csv), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		if len(fields) < nf {
			warns = append(warns, fmt.Sprintf("nvidia-smi: unparseable line %q", line))
			continue
		}
		// Take the trailing fields from the right so a comma in the name is harmless.
		n := len(fields)
		if !withCap {
			fields = append(fields, "")
			n++
		}
		name := strings.TrimSpace(strings.Join(fields[:n-4], ","))
		total, errT := strconv.ParseUint(strings.TrimSpace(fields[n-4]), 10, 64)
		free, errF := strconv.ParseUint(strings.TrimSpace(fields[n-3]), 10, 64)
		if errT != nil || errF != nil {
			warns = append(warns, fmt.Sprintf("nvidia-smi: unparseable memory for %q", name))
			continue
		}
		g := GPU{
			Vendor:        "nvidia",
			Name:          name,
			VRAMBytes:     total * mib,
			FreeVRAMBytes: free * mib,
			Driver:        strings.TrimSpace(fields[n-2]),
			Backend:       "cuda",
			Supported:     true,
		}
		capStr := strings.TrimSpace(fields[n-1])
		if v, err := strconv.ParseFloat(capStr, 64); err != nil {
			if withCap {
				warns = append(warns, fmt.Sprintf("nvidia-smi: compute capability unknown for %q, assuming supported", name))
			}
		} else {
			g.ComputeCap = capStr
			if v < minComputeCap {
				g.Backend, g.Supported = "cpu", false
			}
		}
		g.BandwidthGBs, g.BandwidthLowConfidence = LookupGPUBandwidth("nvidia", g.Name, g.VRAMBytes)
		gpus = append(gpus, g)
	}
	return gpus, warns
}

// DetectNvidia queries nvidia-smi. A missing binary yields no GPUs and no
// warnings; any other failure yields no GPUs and one warning. Old drivers
// reject the compute_cap field, so a failed query is retried once without it.
func DetectNvidia(ctx context.Context, run Runner) ([]GPU, []string) {
	out, err := run(ctx, "nvidia-smi",
		"--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
		"--format=csv,noheader,nounits")
	if err == nil {
		return ParseNvidiaSMI(out)
	}
	if errors.Is(err, exec.ErrNotFound) {
		return nil, nil
	}
	out, err = run(ctx, "nvidia-smi",
		"--query-gpu=name,memory.total,memory.free,driver_version",
		"--format=csv,noheader,nounits")
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, nil
		}
		first, _, _ := strings.Cut(strings.TrimSpace(string(out)), "\n")
		return nil, []string{fmt.Sprintf("nvidia-smi failed (%v): %s", err, strings.TrimSpace(first))}
	}
	gpus, warns := parseNvidiaSMI(out, false)
	if len(gpus) > 0 {
		warns = append(warns, "nvidia-smi: compute capability unavailable (old driver), assuming supported")
	}
	return gpus, warns
}
