package hw

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestAppleGPUShare(t *testing.T) {
	cases := []struct {
		ram  uint64
		want uint64
	}{
		{16 * gib, 16 * gib * 66 / 100},
		{36 * gib, 36 * gib * 66 / 100},
		{64 * gib, 64 * gib * 75 / 100},
	}
	for _, c := range cases {
		if got := AppleGPUShare(c.ram); got != c.want {
			t.Errorf("AppleGPUShare(%d) = %d, want %d", c.ram, got, c.want)
		}
	}
}

func TestParseAppleDisplaysM2Pro(t *testing.T) {
	js, err := os.ReadFile("testdata/system_profiler-m2pro.json")
	if err != nil {
		t.Fatal(err)
	}
	gpus, warns := ParseAppleDisplays(js, "Apple M2 Pro", 32*gib)
	if len(gpus) != 1 || len(warns) != 0 {
		t.Fatalf("gpus=%v warns=%v", gpus, warns)
	}
	g := gpus[0]
	if g.Vendor != "apple" || g.Backend != "metal" || !g.Supported || g.Name != "Apple M2 Pro" {
		t.Errorf("unexpected gpu %+v", g)
	}
	if g.BandwidthGBs != 200 || g.BandwidthLowConfidence {
		t.Errorf("bandwidth %v low=%v", g.BandwidthGBs, g.BandwidthLowConfidence)
	}
	if g.VRAMBytes != AppleGPUShare(32*gib) {
		t.Errorf("VRAM %d", g.VRAMBytes)
	}
}

func TestParseAppleDisplaysUnknownChip(t *testing.T) {
	js, _ := os.ReadFile("testdata/system_profiler-m2pro.json")
	gpus, warns := ParseAppleDisplays(js, "Apple M99 Ultra", 64*gib)
	if len(gpus) != 1 || !gpus[0].Supported || !gpus[0].BandwidthLowConfidence || gpus[0].BandwidthGBs != 100 {
		t.Fatalf("gpus=%+v", gpus)
	}
	if len(warns) != 1 {
		t.Errorf("warns=%v", warns)
	}
}

func TestParseAppleDisplaysIntelMac(t *testing.T) {
	js, err := os.ReadFile("testdata/system_profiler-intel.json")
	if err != nil {
		t.Fatal(err)
	}
	gpus, warns := ParseAppleDisplays(js, "Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz", 32*gib)
	if len(gpus) != 2 {
		t.Fatalf("gpus=%+v", gpus)
	}
	for _, g := range gpus {
		if g.Supported || g.Backend != "cpu" {
			t.Errorf("want unsupported cpu: %+v", g)
		}
	}
	if gpus[1].Name != "AMD Radeon Pro 5500M" || gpus[1].VRAMBytes != 8*gib {
		t.Errorf("discrete gpu %+v", gpus[1])
	}
	if len(warns) != 1 || !strings.Contains(warns[0], "Intel Mac") {
		t.Errorf("warns=%v", warns)
	}
}

func TestParseAppleDisplaysBadJSON(t *testing.T) {
	gpus, warns := ParseAppleDisplays([]byte("nope"), "Apple M2 Pro", 32*gib)
	if len(gpus) != 0 || len(warns) != 1 {
		t.Errorf("gpus=%v warns=%v", gpus, warns)
	}
}

func TestDetectApple(t *testing.T) {
	js, _ := os.ReadFile("testdata/system_profiler-m2pro.json")
	run := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		switch {
		case name == "sysctl" && args[1] == "hw.memsize":
			return []byte("34359738368\n"), nil
		case name == "sysctl":
			return []byte("Apple M2 Pro\n"), nil
		}
		return js, nil
	}
	gpus, ram, chip, warns := DetectApple(context.Background(), run)
	if len(gpus) != 1 || ram != 32*gib || chip != "Apple M2 Pro" || len(warns) != 0 {
		t.Errorf("gpus=%v ram=%d chip=%q warns=%v", gpus, ram, chip, warns)
	}
}
