package hw

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// appleLargeRAMThreshold is the RAM size above which macOS lets the GPU wire
// a larger share of unified memory.
const appleLargeRAMThreshold = 36 * gib

// appleFallbackBandwidthGBs is used when the chip is not in the Apple table.
const appleFallbackBandwidthGBs = 100

// AppleGPUShare returns the unified memory the GPU can use by default:
// 75% when RAM is above 36 GiB, otherwise 66%.
func AppleGPUShare(ramBytes uint64) uint64 {
	if ramBytes > appleLargeRAMThreshold {
		return ramBytes * 75 / 100
	}
	return ramBytes * 66 / 100
}

type spDisplays struct {
	Items []struct {
		Model      string `json:"sppci_model"`
		VRAM       string `json:"spdisplays_vram"`
		VRAMShared string `json:"spdisplays_vram_shared"`
	} `json:"SPDisplaysDataType"`
}

// parseSPVRAM parses values such as "8 GB" or "1536 MB". Returns 0 if unknown.
func parseSPVRAM(s string) uint64 {
	f := strings.Fields(s)
	if len(f) != 2 {
		return 0
	}
	n, err := strconv.ParseUint(f[0], 10, 64)
	if err != nil {
		return 0
	}
	switch strings.ToUpper(f[1]) {
	case "GB":
		return n * gib
	case "MB":
		return n * mib
	}
	return 0
}

// buildAppleSilicon builds the Metal GPU from the chip name and RAM alone.
// An empty chip stands for an Apple Silicon Mac whose brand string could not
// be read.
func buildAppleSilicon(chip string, ramBytes uint64) ([]GPU, []string) {
	name := chip
	if name == "" {
		name = "Apple Silicon (unknown chip)"
	}
	g := GPU{
		Vendor:    "apple",
		Name:      name,
		VRAMBytes: AppleGPUShare(ramBytes),
		Backend:   "metal",
		Supported: true,
	}
	g.FreeVRAMBytes = g.VRAMBytes
	var warns []string
	bw, ok := LookupAppleBandwidth(chip)
	if !ok {
		bw = appleFallbackBandwidthGBs
		g.BandwidthLowConfidence = true
		warns = append(warns, fmt.Sprintf("unknown Apple chip %q, assuming %d GB/s memory bandwidth", chip, appleFallbackBandwidthGBs))
	}
	g.BandwidthGBs = bw
	return []GPU{g}, warns
}

// ParseAppleDisplays turns `system_profiler SPDisplaysDataType -json` output
// into GPUs. chip is the CPU brand string; a chip not starting with "Apple"
// means an Intel Mac, where Ollama does not use the GPU.
func ParseAppleDisplays(js []byte, chip string, ramBytes uint64) ([]GPU, []string) {
	var sp spDisplays
	if err := json.Unmarshal(js, &sp); err != nil {
		return nil, []string{fmt.Sprintf("system_profiler: unparseable output (%v)", err)}
	}
	if strings.HasPrefix(chip, "Apple") {
		return buildAppleSilicon(chip, ramBytes)
	}

	var gpus []GPU
	for _, it := range sp.Items {
		vram := parseSPVRAM(it.VRAM)
		if vram == 0 {
			vram = parseSPVRAM(it.VRAMShared)
		}
		gpus = append(gpus, GPU{
			Name:      it.Model,
			VRAMBytes: vram,
			Backend:   "cpu",
			Supported: false,
		})
		g := &gpus[len(gpus)-1]
		g.FreeVRAMBytes = g.VRAMBytes
		switch l := strings.ToLower(it.Model); {
		case strings.Contains(l, "amd") || strings.Contains(l, "radeon"):
			g.Vendor = "amd"
		case strings.Contains(l, "intel"):
			g.Vendor = "intel"
		}
	}
	var warns []string
	if len(gpus) > 0 {
		warns = []string{"Intel Mac detected: Ollama does not use the GPU on Intel Macs, inference runs on CPU"}
	}
	return gpus, warns
}

// DetectApple queries sysctl, and system_profiler on Intel Macs only. Apple
// Silicon GPUs are built from the chip name and RAM alone. It returns GPUs, total RAM,
// the CPU brand string and warnings. Failures become warnings; a missing
// binary (non-macOS host) yields zero values without a warning.
func DetectApple(ctx context.Context, run Runner, goarch string) ([]GPU, uint64, string, []string) {
	var warns []string
	var ram uint64
	if out, err := run(ctx, "sysctl", "-n", "hw.memsize"); err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, 0, "", nil
		}
		warns = append(warns, fmt.Sprintf("sysctl hw.memsize failed (%v)", err))
	} else if v, perr := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64); perr != nil {
		warns = append(warns, fmt.Sprintf("sysctl hw.memsize: unparseable output %q", strings.TrimSpace(string(out))))
	} else {
		ram = v
	}

	var chip string
	if out, err := run(ctx, "sysctl", "-n", "machdep.cpu.brand_string"); err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, ram, "", warns
		}
		warns = append(warns, fmt.Sprintf("sysctl machdep.cpu.brand_string failed (%v)", err))
	} else {
		chip = strings.TrimSpace(string(out))
	}

	if strings.HasPrefix(chip, "Apple") || (chip == "" && goarch == "arm64") {
		g, w := buildAppleSilicon(chip, ram)
		return g, ram, chip, append(warns, w...)
	}

	out, err := run(ctx, "system_profiler", "SPDisplaysDataType", "-json")
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, ram, chip, warns
		}
		first, _, _ := strings.Cut(strings.TrimSpace(string(out)), "\n")
		return nil, ram, chip, append(warns, fmt.Sprintf("system_profiler failed (%v): %s", err, strings.TrimSpace(first)))
	}
	gpus, w := ParseAppleDisplays(out, chip, ram)
	return gpus, ram, chip, append(warns, w...)
}
