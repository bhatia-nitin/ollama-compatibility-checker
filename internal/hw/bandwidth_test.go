package hw

import "testing"

func TestLookupGPUBandwidth(t *testing.T) {
	tests := []struct {
		vendor, name string
		vram         uint64
		want         float64
		low          bool
	}{
		{"nvidia", "NVIDIA GeForce RTX 4090", 24 << 30, 1008, false},
		{"nvidia", "NVIDIA GeForce RTX 3060", 12 << 30, 360, false},
		{"nvidia", "NVIDIA GeForce RTX 3060 Ti", 8 << 30, 448, false},
		{"nvidia", "NVIDIA GeForce RTX 4070", 12 << 30, 504, false},
		{"nvidia", "NVIDIA GeForce RTX 4070 Ti SUPER", 16 << 30, 672, false},
		{"nvidia", "NVIDIA   GeForce(R) RTX 4070  Ti Super", 16 << 30, 672, false},
		{"amd", "AMD Radeon RX 7900 XTX", 24 << 30, 960, false},
		{"amd", "AMD Radeon RX 7900 XT", 20 << 30, 800, false},
		{"intel", "Intel(R) Arc(TM) A770", 16 << 30, 560, false},
		{"nvidia", "NVIDIA Future 9999", 16 << 30, 500, true},
		{"nvidia", "unknown", 8 << 30, 300, true},
		{"nvidia", "unknown", 12 << 30, 450, true},
		{"nvidia", "unknown", 24 << 30, 900, true},
		{"nvidia", "unknown", 48 << 30, 1500, true},
		{"amd", "AMD Future", 8 << 30, 240, true},
		{"intel", "Intel Future", 16 << 30, 400, true},
	}
	for _, tc := range tests {
		got, low := LookupGPUBandwidth(tc.vendor, tc.name, tc.vram)
		if got != tc.want || low != tc.low {
			t.Errorf("%s %q: got (%v,%v), want (%v,%v)", tc.vendor, tc.name, got, low, tc.want, tc.low)
		}
	}
}

func TestLookupAppleBandwidth(t *testing.T) {
	tests := []struct {
		chip string
		want float64
		ok   bool
	}{
		{"Apple M1", 68.25, true},
		{"Apple M2 Pro", 200, true},
		{"Apple M2 Max", 400, true},
		{"Apple M3 Pro", 150, true},
		{"Apple M4 Max", 546, true},
		{"Apple M4", 120, true},
		{"Intel(R) Core(TM) i9", 0, false},
	}
	for _, tc := range tests {
		got, ok := LookupAppleBandwidth(tc.chip)
		if got != tc.want || ok != tc.ok {
			t.Errorf("%q: got (%v,%v), want (%v,%v)", tc.chip, got, ok, tc.want, tc.ok)
		}
	}
}

func TestRAMBandwidthFallback(t *testing.T) {
	if got := RAMBandwidthFallback("linux", "amd64"); got != 50 {
		t.Errorf("got %v, want 50", got)
	}
}
