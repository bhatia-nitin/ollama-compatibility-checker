package hw

import (
	"context"
	"os/exec"
	"strings"
	"testing"
)

func TestParseRocmSMI(t *testing.T) {
	gpus, warns := ParseRocmSMI(readFixture(t, "rocm-smi.json"))
	if len(gpus) != 1 || len(warns) != 0 {
		t.Fatalf("gpus=%d warns=%v", len(gpus), warns)
	}
	g := gpus[0]
	if g.VRAMBytes != 24*1024*1024*1024 {
		t.Errorf("vram=%d", g.VRAMBytes)
	}
	if g.Vendor != "amd" || g.Backend != "rocm" || !g.Supported || g.ComputeCap != "gfx1100" {
		t.Errorf("got %+v", g)
	}
	if g.BandwidthGBs != 960 || g.BandwidthLowConfidence {
		t.Errorf("bandwidth=%v low=%v", g.BandwidthGBs, g.BandwidthLowConfidence)
	}
	if g.FreeVRAMBytes != 24*1024*1024*1024-1073741824 {
		t.Errorf("free=%d", g.FreeVRAMBytes)
	}
}

func TestAMDUnsupportedTarget(t *testing.T) {
	js := `{"card0":{"Card Series":"AMD Radeon Graphics","GFX Version":"gfx1035","VRAM Total Memory (B)":"536870912"}}`
	gpus, _ := ParseRocmSMI([]byte(js))
	if len(gpus) != 1 {
		t.Fatalf("gpus=%d", len(gpus))
	}
	if gpus[0].Supported || gpus[0].Backend != "cpu" {
		t.Errorf("got %+v", gpus[0])
	}
}

func TestParseRocmSMISkipsCardWithoutVRAM(t *testing.T) {
	js := `{"card0":{"Card Series":"X","GFX Version":"gfx1100"},"system":{"Driver version":"6.3"}}`
	gpus, warns := ParseRocmSMI([]byte(js))
	if len(gpus) != 0 || len(warns) != 1 || !strings.Contains(warns[0], "card0") {
		t.Errorf("gpus=%v warns=%v", gpus, warns)
	}
}

func TestDetectAMDSysfsFallback(t *testing.T) {
	run := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		return nil, exec.ErrNotFound
	}
	gpus, warns := DetectAMD(context.Background(), run, "testdata/sysfs")
	if len(gpus) != 1 {
		t.Fatalf("gpus=%d warns=%v", len(gpus), warns)
	}
	if gpus[0].VRAMBytes != 17163091968 || gpus[0].Vendor != "amd" {
		t.Errorf("got %+v", gpus[0])
	}
	if !gpus[0].Supported || gpus[0].Backend != "rocm" {
		t.Errorf("RX 6800 XT should be supported: %+v", gpus[0])
	}
	if len(warns) == 0 || !strings.Contains(warns[0], "rocm-smi not found") {
		t.Errorf("warns=%v", warns)
	}
}

func TestDetectAMDSkipsSysfsOnSuccess(t *testing.T) {
	run := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		return readFixture(t, "rocm-smi.json"), nil
	}
	gpus, warns := DetectAMD(context.Background(), run, "testdata/sysfs")
	if len(gpus) != 1 || gpus[0].Name != "Radeon RX 7900 XTX" || len(warns) != 0 {
		t.Errorf("gpus=%+v warns=%v", gpus, warns)
	}
}
