package hw

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// rocmSupportedTargets lists the gfx targets Ollama's ROCm backend supports.
var rocmSupportedTargets = map[string]bool{
	"gfx900": true, "gfx906": true, "gfx908": true, "gfx90a": true,
	"gfx940": true, "gfx941": true, "gfx942": true,
	"gfx1030": true, "gfx1100": true, "gfx1101": true, "gfx1102": true,
	"gfx1200": true, "gfx1201": true,
}

var amdCardDir = regexp.MustCompile(`^card[0-9]+$`)

// jsonString renders a JSON scalar as a trimmed string.
func jsonString(v any) string {
	switch x := v.(type) {
	case string:
		return strings.TrimSpace(x)
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	}
	return ""
}

// findKey returns the value of the first key (in sorted order) whose
// lowercased name contains every want substring and none of the avoid ones.
func findKey(card map[string]any, want []string, avoid ...string) string {
	keys := make([]string, 0, len(card))
	for k := range card {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		lk := strings.ToLower(k)
		ok := true
		for _, w := range want {
			ok = ok && strings.Contains(lk, w)
		}
		for _, a := range avoid {
			ok = ok && !strings.Contains(lk, a)
		}
		if ok {
			if s := jsonString(card[k]); s != "" {
				return s
			}
		}
	}
	return ""
}

// ParseRocmSMI parses `rocm-smi --showproductname --showmeminfo vram --showhw --json`.
// Key names vary between ROCm versions, so fields are found by key-name match.
// Cards without a parseable VRAM total are skipped with a warning.
func ParseRocmSMI(js []byte) ([]GPU, []string) {
	var top map[string]map[string]any
	if err := json.Unmarshal(js, &top); err != nil {
		return nil, []string{fmt.Sprintf("rocm-smi: unparseable JSON: %v", err)}
	}
	ids := make([]string, 0, len(top))
	for id := range top {
		if strings.HasPrefix(strings.ToLower(id), "card") {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)

	var gpus []GPU
	var warns []string
	for _, id := range ids {
		card := top[id]
		name := findKey(card, []string{"card", "series"})
		if name == "" {
			name = findKey(card, []string{"device", "name"})
		}
		if name == "" {
			name = "AMD GPU"
		}
		total, err := strconv.ParseUint(findKey(card, []string{"vram", "total", "memory"}, "used"), 10, 64)
		if err != nil || total == 0 {
			warns = append(warns, fmt.Sprintf("rocm-smi: no VRAM total for %s, skipping", id))
			continue
		}
		g := GPU{Vendor: "amd", Name: name, VRAMBytes: total}
		if used, err := strconv.ParseUint(findKey(card, []string{"vram", "used"}), 10, 64); err == nil && used <= total {
			g.FreeVRAMBytes = total - used
		}
		g.ComputeCap = strings.ToLower(findKey(card, []string{"gfx"}))
		switch {
		case g.ComputeCap == "":
			g.Backend, g.Supported = "cpu", false
			warns = append(warns, fmt.Sprintf("rocm-smi: gfx target unknown for %q, treating as unsupported", name))
		case rocmSupportedTargets[g.ComputeCap]:
			g.Backend, g.Supported = "rocm", true
		default:
			g.Backend, g.Supported = "cpu", false
		}
		g.BandwidthGBs, g.BandwidthLowConfidence = LookupGPUBandwidth("amd", g.Name, g.VRAMBytes)
		gpus = append(gpus, g)
	}
	return gpus, warns
}

// DetectAMDSysfs reads <root>/sys/class/drm/card*/device for AMD (0x1002)
// devices. The gfx target is unavailable here, so a GPU counts as supported
// only when its name matches the bandwidth table for a known discrete card.
func DetectAMDSysfs(root string) ([]GPU, []string) {
	dirs, _ := filepath.Glob(filepath.Join(root, "sys/class/drm/card*"))
	sort.Strings(dirs)
	var gpus []GPU
	var warns []string
	for _, d := range dirs {
		if !amdCardDir.MatchString(filepath.Base(d)) {
			continue
		}
		dev := filepath.Join(d, "device")
		vendor, err := os.ReadFile(filepath.Join(dev, "vendor"))
		if err != nil || strings.TrimSpace(string(vendor)) != "0x1002" {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dev, "mem_info_vram_total"))
		total, perr := strconv.ParseUint(strings.TrimSpace(string(raw)), 10, 64)
		if err != nil || perr != nil || total == 0 {
			warns = append(warns, fmt.Sprintf("amd sysfs: no VRAM total for %s, skipping", filepath.Base(d)))
			continue
		}
		name := "AMD GPU"
		if b, err := os.ReadFile(filepath.Join(dev, "product_name")); err == nil && strings.TrimSpace(string(b)) != "" {
			name = strings.TrimSpace(string(b))
		}
		g := GPU{Vendor: "amd", Name: name, VRAMBytes: total}
		g.BandwidthGBs, g.BandwidthLowConfidence = LookupGPUBandwidth("amd", name, total)
		if g.BandwidthLowConfidence {
			g.Backend, g.Supported = "cpu", false
			warns = append(warns, fmt.Sprintf("amd sysfs: cannot confirm ROCm support for %q (no gfx target), treating as unsupported", name))
		} else {
			g.Backend, g.Supported = "rocm", true
		}
		gpus = append(gpus, g)
	}
	return gpus, warns
}

// DetectAMD queries rocm-smi, falling back to sysfs when it is missing or
// fails. sysRoot is "/" in production. If rocm-smi succeeds, sysfs is not read.
func DetectAMD(ctx context.Context, run Runner, sysRoot string) ([]GPU, []string) {
	out, err := run(ctx, "rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showhw", "--json")
	var warns []string
	if err == nil {
		return ParseRocmSMI(out)
	}
	if errors.Is(err, exec.ErrNotFound) {
		warns = append(warns, "rocm-smi not found, falling back to sysfs")
	} else {
		first, _, _ := strings.Cut(strings.TrimSpace(string(out)), "\n")
		warns = append(warns, fmt.Sprintf("rocm-smi failed (%v): %s, falling back to sysfs", err, strings.TrimSpace(first)))
	}
	gpus, w := DetectAMDSysfs(sysRoot)
	return gpus, append(warns, w...)
}
