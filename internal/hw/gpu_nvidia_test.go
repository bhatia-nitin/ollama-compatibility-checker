package hw

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestParseNvidiaSMISingle(t *testing.T) {
	gpus, warns := ParseNvidiaSMI(readFixture(t, "nvidia-smi-single.csv"))
	if len(gpus) != 1 || len(warns) != 0 {
		t.Fatalf("gpus=%d warns=%v", len(gpus), warns)
	}
	g := gpus[0]
	if g.VRAMBytes != 12282*1024*1024 || g.FreeVRAMBytes != 11500*1024*1024 {
		t.Errorf("vram=%d free=%d", g.VRAMBytes, g.FreeVRAMBytes)
	}
	if g.Vendor != "nvidia" || g.Backend != "cuda" || !g.Supported {
		t.Errorf("got %+v", g)
	}
	if g.BandwidthGBs != 504 || g.BandwidthLowConfidence {
		t.Errorf("bandwidth=%v low=%v", g.BandwidthGBs, g.BandwidthLowConfidence)
	}
	if g.Driver != "560.94" || g.ComputeCap != "8.9" {
		t.Errorf("driver=%q cap=%q", g.Driver, g.ComputeCap)
	}
}

func TestParseNvidiaSMIDual(t *testing.T) {
	gpus, warns := ParseNvidiaSMI(readFixture(t, "nvidia-smi-dual.csv"))
	if len(gpus) != 2 || len(warns) != 0 {
		t.Fatalf("gpus=%d warns=%v", len(gpus), warns)
	}
	if !gpus[0].Supported || !gpus[1].Supported {
		t.Errorf("cap 5.0 must be supported: %+v", gpus)
	}
}

func TestParseNvidiaSMIOldCard(t *testing.T) {
	gpus, _ := ParseNvidiaSMI([]byte("NVIDIA GeForce GTX 780, 3072, 3000, 470.82, 3.5\n"))
	if len(gpus) != 1 {
		t.Fatalf("gpus=%d", len(gpus))
	}
	if gpus[0].Supported || gpus[0].Backend != "cpu" {
		t.Errorf("got %+v", gpus[0])
	}
}

func TestParseNvidiaSMINoCap(t *testing.T) {
	gpus, warns := ParseNvidiaSMI(readFixture(t, "nvidia-smi-nocap.csv"))
	if len(gpus) != 1 || !gpus[0].Supported || gpus[0].Backend != "cuda" {
		t.Fatalf("gpus=%+v", gpus)
	}
	if len(warns) != 1 || !strings.Contains(warns[0], "compute capability") {
		t.Errorf("warns=%v", warns)
	}
}

func TestDetectNvidiaToolFails(t *testing.T) {
	run := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		return []byte("NVIDIA-SMI has failed\nmore"), errors.New("exit status 9")
	}
	gpus, warns := DetectNvidia(context.Background(), run)
	if len(gpus) != 0 || len(warns) != 1 {
		t.Fatalf("gpus=%d warns=%v", len(gpus), warns)
	}
	if !strings.Contains(warns[0], "NVIDIA-SMI has failed") || strings.Contains(warns[0], "more") {
		t.Errorf("warn=%q", warns[0])
	}
}

func TestDetectNvidiaNotInstalled(t *testing.T) {
	run := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		return nil, &exec.Error{Name: name, Err: exec.ErrNotFound}
	}
	gpus, warns := DetectNvidia(context.Background(), run)
	if len(gpus) != 0 || len(warns) != 0 {
		t.Fatalf("gpus=%d warns=%v", len(gpus), warns)
	}
}

func TestDetectNvidiaOK(t *testing.T) {
	var gotName string
	var gotArgs []string
	run := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		gotName, gotArgs = name, args
		return readFixture(t, "nvidia-smi-single.csv"), nil
	}
	gpus, _ := DetectNvidia(context.Background(), run)
	if len(gpus) != 1 || gotName != "nvidia-smi" {
		t.Fatalf("gpus=%d name=%q", len(gpus), gotName)
	}
	want := "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap"
	if len(gotArgs) != 2 || gotArgs[0] != want || gotArgs[1] != "--format=csv,noheader,nounits" {
		t.Errorf("args=%v", gotArgs)
	}
}

func TestDetectNvidiaOldDriverRetriesWithoutComputeCap(t *testing.T) {
	var queries []string
	run := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		queries = append(queries, args[0])
		if strings.Contains(args[0], "compute_cap") {
			return []byte("Field \"compute_cap\" is not a valid field"), errors.New("exit status 2")
		}
		return []byte("NVIDIA GeForce GTX 1080, 8192, 8000, 390.1\n"), nil
	}
	gpus, warns := DetectNvidia(context.Background(), run)
	if len(queries) != 2 || queries[1] != "--query-gpu=name,memory.total,memory.free,driver_version" {
		t.Fatalf("queries=%v", queries)
	}
	if len(gpus) != 1 || !gpus[0].Supported || gpus[0].Backend != "cuda" || gpus[0].Driver != "390.1" || gpus[0].ComputeCap != "" {
		t.Fatalf("gpus=%+v", gpus)
	}
	if len(warns) != 1 || !strings.Contains(warns[0], "compute capability unavailable") {
		t.Errorf("warns=%v", warns)
	}
}
