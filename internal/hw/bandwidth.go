package hw

import "strings"

// Published memory bandwidth in GB/s, keyed by normalized name substring.
// Entries without a firm spec-sheet figure are omitted on purpose; the
// VRAM-class fallback covers them.
var nvidiaBandwidth = map[string]float64{
	// RTX 20
	"rtx 2060":       336,
	"rtx 2060 super": 448,
	"rtx 2070":       448,
	"rtx 2070 super": 448,
	"rtx 2080":       448,
	"rtx 2080 super": 496,
	"rtx 2080 ti":    616,
	// RTX 30
	"rtx 3050":    224,
	"rtx 3060":    360,
	"rtx 3060 ti": 448,
	"rtx 3070":    448,
	"rtx 3070 ti": 608,
	"rtx 3080":    760,
	"rtx 3080 ti": 912,
	"rtx 3090":    936,
	"rtx 3090 ti": 1008,
	// RTX 40
	"rtx 4060":          272,
	"rtx 4060 ti":       288,
	"rtx 4070":          504,
	"rtx 4070 super":    504,
	"rtx 4070 ti":       504,
	"rtx 4070 ti super": 672,
	"rtx 4080":          717,
	"rtx 4080 super":    736,
	"rtx 4090":          1008,
	// RTX 50
	"rtx 5050":    320,
	"rtx 5060":    448,
	"rtx 5060 ti": 448,
	"rtx 5070":    672,
	"rtx 5070 ti": 896,
	"rtx 5080":    960,
	"rtx 5090":    1792,
	// Datacenter
	"a100":           1555,
	"a100 80gb":      2039,
	"a100-sxm4-80gb": 2039,
	"h100 pcie":      2000,
	"h100 80gb hbm3": 3350,
	"l4":             300,
	"t4":             320,
}

// nvidiaLaptopBandwidth holds NVIDIA's reference memory configuration for
// laptop GPUs (bus width x data rate). Laptop parts use different memory
// from the desktop card with the same number, so the desktop table does not
// apply. OEMs can clock memory slightly differently.
var nvidiaLaptopBandwidth = map[string]float64{
	"rtx 2060":    336, // 192-bit, 14 Gbps
	"rtx 2070":    448, // 256-bit, 14 Gbps
	"rtx 2080":    448,
	"rtx 3050":    192, // 128-bit, 12 Gbps
	"rtx 3050 ti": 192,
	"rtx 3060":    336, // 192-bit, 14 Gbps
	"rtx 3070":    448, // 256-bit, 14 Gbps
	"rtx 3070 ti": 448,
	"rtx 3080":    448,
	"rtx 3080 ti": 512, // 256-bit, 16 Gbps
	"rtx 4050":    192, // 96-bit, 16 Gbps
	"rtx 4060":    256, // 128-bit, 16 Gbps
	"rtx 4070":    256,
	"rtx 4080":    432, // 192-bit, 18 Gbps
	"rtx 4090":    576, // 256-bit, 18 Gbps
}

var amdBandwidth = map[string]float64{
	// RX 6000
	"rx 6500 xt": 144,
	"rx 6600":    224,
	"rx 6600 xt": 256,
	"rx 6650 xt": 280,
	"rx 6700 xt": 384,
	"rx 6750 xt": 432,
	"rx 6800":    512,
	"rx 6800 xt": 512,
	"rx 6900 xt": 512,
	"rx 6950 xt": 576,
	// RX 7000
	"rx 7600":     288,
	"rx 7600 xt":  288,
	"rx 7700 xt":  432,
	"rx 7800 xt":  624,
	"rx 7900 gre": 576,
	"rx 7900 xt":  800,
	"rx 7900 xtx": 960,
	// RX 9000
	"rx 9060 xt": 322,
	"rx 9070":    624,
	"rx 9070 xt": 640,
	// Instinct
	"mi100":  1229,
	"mi210":  1638,
	"mi250x": 3277,
	"mi300x": 5300,
}

var intelBandwidth = map[string]float64{
	"arc a380": 186,
	"arc a580": 512,
	"arc a750": 512,
	"arc a770": 560,
	"arc b570": 380,
	"arc b580": 456,
}

var appleBandwidth = map[string]float64{
	"apple m1":       68.25,
	"apple m1 pro":   200,
	"apple m1 max":   400,
	"apple m1 ultra": 800,
	"apple m2":       100,
	"apple m2 pro":   200,
	"apple m2 max":   400,
	"apple m2 ultra": 800,
	"apple m3":       100,
	"apple m3 pro":   150,
	"apple m3 max":   400,
	"apple m3 ultra": 819,
	"apple m4":       120,
	"apple m4 pro":   273,
	"apple m4 max":   546,
}

const gib = uint64(1) << 30

// normalizeName lowercases, strips (r)/(tm) and collapses whitespace.
func normalizeName(s string) string {
	s = strings.ToLower(s)
	s = strings.ReplaceAll(s, "(r)", "")
	s = strings.ReplaceAll(s, "(tm)", "")
	return strings.Join(strings.Fields(s), " ")
}

// longestMatch returns the value of the longest table key contained in name.
func longestMatch(table map[string]float64, name string) (float64, bool) {
	n := normalizeName(name)
	bestLen := 0
	var best float64
	for k, v := range table {
		if len(k) > bestLen && containsToken(n, k) {
			bestLen, best = len(k), v
		}
	}
	return best, bestLen > 0
}

// containsToken reports whether key occurs in name with non-alphanumeric
// characters (or string ends) on both sides, so "l4" does not match "l40s".
func containsToken(name, key string) bool {
	for off := 0; ; {
		i := strings.Index(name[off:], key)
		if i < 0 {
			return false
		}
		start, end := off+i, off+i+len(key)
		if (start == 0 || !isAlnum(name[start-1])) && (end == len(name) || !isAlnum(name[end])) {
			return true
		}
		off = start + 1
	}
}

func isAlnum(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= '0' && c <= '9'
}

// isMobileName reports laptop/mobile variants, which have far lower
// bandwidth than the desktop part sharing the model number.
func isMobileName(name string) bool {
	n := normalizeName(name)
	return strings.Contains(n, "laptop") || strings.Contains(n, "mobile") || strings.Contains(n, "max-q")
}

// vramClassBandwidth is the fallback estimate by VRAM size class.
func vramClassBandwidth(vendor string, vramBytes uint64) float64 {
	var bw float64
	switch {
	case vramBytes <= 8*gib:
		bw = 300
	case vramBytes <= 12*gib:
		bw = 450
	case vramBytes <= 16*gib:
		bw = 500
	case vramBytes <= 24*gib:
		bw = 900
	default:
		bw = 1500
	}
	if vendor == "amd" || vendor == "intel" {
		bw *= 0.8
	}
	return bw
}

// LookupGPUBandwidth returns memory bandwidth in GB/s for a discrete GPU.
// lowConfidence is true when the value came from the VRAM-class fallback.
func LookupGPUBandwidth(vendor, name string, vramBytes uint64) (gbs float64, lowConfidence bool) {
	var table map[string]float64
	switch vendor {
	case "nvidia":
		table = nvidiaBandwidth
	case "amd":
		table = amdBandwidth
	case "intel":
		table = intelBandwidth
	case "apple":
		table = appleBandwidth
	}
	if isMobileName(name) && vendor == "nvidia" {
		if v, ok := longestMatch(nvidiaLaptopBandwidth, name); ok {
			return v, false
		}
	}
	if v, ok := longestMatch(table, name); ok {
		if isMobileName(name) { // laptop part not in a laptop table: rough guess
			return v * 0.5, true
		}
		return v, false
	}
	return vramClassBandwidth(vendor, vramBytes), true
}

// LookupAppleBandwidth returns unified memory bandwidth for an Apple chip
// string such as "Apple M2 Pro".
func LookupAppleBandwidth(chip string) (gbs float64, ok bool) {
	return longestMatch(appleBandwidth, chip)
}

// RAMBandwidthFallback returns a conservative system RAM bandwidth in GB/s.
// Apple Silicon uses the Apple table instead, so it is not special-cased.
func RAMBandwidthFallback(os, arch string) float64 {
	return 50
}
