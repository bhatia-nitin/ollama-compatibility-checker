package hw

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestMergeGPUsPoolsSameVendor(t *testing.T) {
	in := []GPU{
		{Vendor: "nvidia", Name: "NVIDIA GeForce RTX 4070", VRAMBytes: 12 * gib, FreeVRAMBytes: 11 * gib, BandwidthGBs: 504, Backend: "cuda", Supported: true},
		{Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3060", VRAMBytes: 8 * gib, FreeVRAMBytes: 7 * gib, BandwidthGBs: 272, BandwidthLowConfidence: true, Backend: "cuda", Supported: true},
	}
	out, warns := MergeGPUs(in)
	if len(out) != 1 || len(warns) != 0 {
		t.Fatalf("got %d gpus, warns %v", len(out), warns)
	}
	g := out[0]
	if g.Name != "2x GPU (NVIDIA GeForce RTX 4070 + NVIDIA GeForce RTX 3060)" {
		t.Errorf("name %q", g.Name)
	}
	if g.VRAMBytes != 20*gib || g.FreeVRAMBytes != 18*gib || g.BandwidthGBs != 272 || !g.BandwidthLowConfidence || !g.Supported {
		t.Errorf("pooled %+v", g)
	}
}

func TestMergeGPUsPooledFreeZeroWhenAnyUnknown(t *testing.T) {
	in := []GPU{
		{Vendor: "nvidia", Name: "A", VRAMBytes: 8 * gib, FreeVRAMBytes: 7 * gib, BandwidthGBs: 100, Supported: true},
		{Vendor: "nvidia", Name: "B", VRAMBytes: 8 * gib, BandwidthGBs: 100, Supported: true},
	}
	out, _ := MergeGPUs(in)
	if out[0].FreeVRAMBytes != 0 {
		t.Errorf("free %d", out[0].FreeVRAMBytes)
	}
}

func TestMergeGPUsMixedVendors(t *testing.T) {
	in := []GPU{
		{Vendor: "amd", Name: "AMD RX 7600", VRAMBytes: 8 * gib, Supported: true},
		{Vendor: "nvidia", Name: "NVIDIA RTX 4070", VRAMBytes: 12 * gib, Supported: true},
	}
	out, warns := MergeGPUs(in)
	if len(out) != 1 || out[0].Vendor != "nvidia" {
		t.Fatalf("out %+v", out)
	}
	if len(warns) != 1 || !strings.Contains(warns[0], "mixed vendors") {
		t.Errorf("warns %v", warns)
	}
}

func TestMergeGPUsUnsupportedKeptNotPooled(t *testing.T) {
	in := []GPU{
		{Vendor: "intel", Name: "Intel UHD", VRAMBytes: 1 * gib},
		{Vendor: "nvidia", Name: "N1", VRAMBytes: 8 * gib, Supported: true},
		{Vendor: "nvidia", Name: "N2", VRAMBytes: 8 * gib, Supported: true},
		{Vendor: "amd", Name: "Old AMD", VRAMBytes: 4 * gib},
	}
	out, _ := MergeGPUs(in)
	if len(out) != 3 || !out[0].Supported || out[0].VRAMBytes != 16*gib || out[1].Name != "Intel UHD" || out[2].Name != "Old AMD" {
		t.Fatalf("out %+v", out)
	}
}

func notFoundRunner(context.Context, string, ...string) ([]byte, error) {
	return nil, exec.ErrNotFound
}

func TestDetectNoGPU(t *testing.T) {
	hw := Detect(context.Background(), Probes{Run: notFoundRunner, SysRoot: t.TempDir(), GOOS: "linux", GOARCH: "amd64"})
	if len(hw.GPUs) != 0 {
		t.Errorf("gpus %+v", hw.GPUs)
	}
	if hw.OS != "linux" || hw.Arch != "amd64" || hw.RAMBandwidthGBs <= 0 {
		t.Errorf("hw %+v", hw)
	}
}

func TestDetectLinuxFixture(t *testing.T) {
	lspci, err := os.ReadFile("testdata/lspci-mm.txt")
	if err != nil {
		t.Fatal(err)
	}
	run := func(_ context.Context, name string, _ ...string) ([]byte, error) {
		if name == "lspci" {
			return lspci, nil
		}
		return nil, exec.ErrNotFound
	}
	hw := Detect(context.Background(), Probes{Run: run, SysRoot: "testdata/linux", GOOS: "linux", GOARCH: "amd64"})
	if hw.RAMBytes != 32768000*1024 {
		t.Errorf("ram %d", hw.RAMBytes)
	}
	c := hw.CPU
	if c.Model != "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz" || c.Threads != 4 || c.Cores != 2 || !c.AVX2 || !c.AVX512 {
		t.Errorf("cpu %+v", c)
	}
	if len(hw.GPUs) != 1 || hw.GPUs[0].Vendor != "intel" || hw.GPUs[0].Supported || hw.GPUs[0].Name != "Intel Corporation UHD Graphics 630" {
		t.Errorf("gpus %+v", hw.GPUs)
	}
}

func TestDetectDarwinApple(t *testing.T) {
	sp, _ := os.ReadFile("testdata/system_profiler-m2pro.json")
	run := func(_ context.Context, name string, args ...string) ([]byte, error) {
		switch strings.Join(append([]string{name}, args...), " ") {
		case "sysctl -n hw.memsize":
			return []byte("17179869184\n"), nil
		case "sysctl -n machdep.cpu.brand_string":
			return []byte("Apple M2 Pro\n"), nil
		case "sysctl -n hw.physicalcpu":
			return []byte("10\n"), nil
		case "sysctl -n hw.logicalcpu":
			return []byte("10\n"), nil
		case "system_profiler SPDisplaysDataType -json":
			return sp, nil
		}
		return nil, exec.ErrNotFound
	}
	hw := Detect(context.Background(), Probes{Run: run, GOOS: "darwin", GOARCH: "arm64"})
	if !hw.UnifiedMemory || hw.RAMBytes != 16*gib || len(hw.GPUs) != 1 || hw.RAMBandwidthGBs != hw.GPUs[0].BandwidthGBs || hw.CPU.Cores != 10 {
		t.Errorf("hw %+v", hw)
	}
}
