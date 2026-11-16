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
	var gpus []GPU
	var warns []string
	for _, line := range strings.Split(string(csv), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		if len(fields) < 5 {
			warns = append(warns, fmt.Sprintf("nvidia-smi: unparseable line %q", line))
			continue
		}
		// Take the last four fields from the right so a comma in the name is harmless.
		n := len(fields)
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
			warns = append(warns, fmt.Sprintf("nvidia-smi: compute capability unknown for %q, assuming supported", name))
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
// warnings; any other failure yields no GPUs and one warning.
func DetectNvidia(ctx context.Context, run Runner) ([]GPU, []string) {
	out, err := run(ctx, "nvidia-smi",
		"--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
		"--format=csv,noheader,nounits")
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, nil
		}
		first, _, _ := strings.Cut(strings.TrimSpace(string(out)), "\n")
		return nil, []string{fmt.Sprintf("nvidia-smi failed (%v): %s", err, strings.TrimSpace(first))}
	}
	return ParseNvidiaSMI(out)
}
