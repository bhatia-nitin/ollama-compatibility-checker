package hw

import (
	"fmt"
	"regexp"
	"strings"
)

// RegAdapter is one display adapter entry read from the Windows registry
// display class key.
type RegAdapter struct {
	DriverDesc    string
	QwMemorySize  uint64
	DriverVersion string
}

// winAMDSupported matches the Radeon models Ollama supports on Windows:
// RX 6800, 6800 XT, 6900 XT, 6950 XT, and all RX 7000 and RX 9000 parts.
var winAMDSupported = regexp.MustCompile(`\brx\s*(?:6800|6900|6950|[79][0-9]{3})m?\b`)

// decodeLEUint decodes a 4 or 8 byte little endian REG_BINARY value.
func decodeLEUint(b []byte) (uint64, bool) {
	if len(b) != 4 && len(b) != 8 {
		return 0, false
	}
	var v uint64
	for i := len(b) - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v, true
}

// ParseWindowsAdapters converts registry adapter entries into GPUs.
// NVIDIA adapters are skipped (nvidia-smi covers them), as are Microsoft
// basic adapters and unrecognized vendors.
func ParseWindowsAdapters(adapters []RegAdapter) ([]GPU, []string) {
	var gpus []GPU
	var warns []string
	for _, a := range adapters {
		name := strings.TrimSpace(a.DriverDesc)
		n := strings.ToLower(name)
		switch {
		case n == "", strings.Contains(n, "microsoft basic"), strings.Contains(n, "nvidia"):
			continue
		}
		g := GPU{Name: name, VRAMBytes: a.QwMemorySize, Driver: a.DriverVersion}
		switch {
		case strings.Contains(n, "amd") || strings.Contains(n, "radeon"):
			g.Vendor = "amd"
			if winAMDSupported.MatchString(normalizeName(name)) {
				g.Backend, g.Supported = "rocm", true
			} else {
				g.Backend, g.Supported = "cpu", false
				warns = append(warns, fmt.Sprintf("windows registry: %q is not on Ollama's Windows ROCm support list, treating as unsupported", name))
			}
		case strings.Contains(n, "intel"):
			g.Vendor = "intel"
			g.Backend, g.Supported = "cpu", false
			warns = append(warns, fmt.Sprintf("windows registry: Intel GPU %q has no supported Ollama backend (Vulkan is experimental), treating as unsupported", name))
		default:
			continue
		}
		g.BandwidthGBs, g.BandwidthLowConfidence = LookupGPUBandwidth(g.Vendor, name, g.VRAMBytes)
		gpus = append(gpus, g)
	}
	return gpus, warns
}
