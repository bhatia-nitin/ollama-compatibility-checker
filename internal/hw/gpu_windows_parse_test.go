package hw

import (
	"strings"
	"testing"
)

func TestParseWindowsAdapters(t *testing.T) {
	gpus, warns := ParseWindowsAdapters([]RegAdapter{
		{DriverDesc: "AMD Radeon RX 7800 XT", QwMemorySize: 16 * gib, DriverVersion: "31.0.1"},
		{DriverDesc: "Microsoft Basic Display Adapter"},
		{DriverDesc: "Intel(R) UHD Graphics 770", QwMemorySize: 128 << 20},
		{DriverDesc: "NVIDIA GeForce RTX 4090", QwMemorySize: 24 * gib},
		{DriverDesc: "Some Other Vendor GPU", QwMemorySize: gib},
	})
	if len(gpus) != 2 {
		t.Fatalf("got %d gpus: %+v", len(gpus), gpus)
	}
	amd := gpus[0]
	if amd.Vendor != "amd" || amd.VRAMBytes != 16*gib || amd.Backend != "rocm" || !amd.Supported || amd.Driver != "31.0.1" {
		t.Errorf("amd wrong: %+v", amd)
	}
	if amd.BandwidthGBs <= 0 {
		t.Errorf("amd bandwidth not set: %+v", amd)
	}
	intel := gpus[1]
	if intel.Vendor != "intel" || intel.Backend != "cpu" || intel.Supported {
		t.Errorf("intel wrong: %+v", intel)
	}
	if len(warns) != 1 || !strings.Contains(warns[0], "Vulkan") {
		t.Errorf("warns = %v", warns)
	}
}

func TestParseWindowsAdaptersAMDSupport(t *testing.T) {
	cases := map[string]bool{
		"AMD Radeon RX 6800":     true,
		"AMD Radeon RX 6800 XT":  true,
		"AMD Radeon RX 6900 XT":  true,
		"AMD Radeon RX 6950 XT":  true,
		"AMD Radeon RX 7600":     true,
		"AMD Radeon RX 7900 XTX": true,
		"AMD Radeon RX 9070 XT":  true,
		"AMD Radeon RX 6700 XT":  false,
		"AMD Radeon RX 6600":     false,
		"AMD Radeon RX 580":      false,
	}
	for name, want := range cases {
		gpus, warns := ParseWindowsAdapters([]RegAdapter{{DriverDesc: name, QwMemorySize: 8 * gib}})
		if len(gpus) != 1 {
			t.Fatalf("%s: got %d gpus", name, len(gpus))
		}
		g := gpus[0]
		if g.Supported != want {
			t.Errorf("%s: Supported=%v want %v", name, g.Supported, want)
		}
		if want && (g.Backend != "rocm" || len(warns) != 0) {
			t.Errorf("%s: backend=%s warns=%v", name, g.Backend, warns)
		}
		if !want && (g.Backend != "cpu" || len(warns) != 1) {
			t.Errorf("%s: backend=%s warns=%v", name, g.Backend, warns)
		}
	}
}

func TestParseWindowsAdaptersSkips(t *testing.T) {
	gpus, warns := ParseWindowsAdapters([]RegAdapter{
		{DriverDesc: ""},
		{DriverDesc: "Microsoft Basic Render Driver", QwMemorySize: gib},
		{DriverDesc: "NVIDIA RTX A2000", QwMemorySize: 12 * gib},
	})
	if len(gpus) != 0 || len(warns) != 0 {
		t.Errorf("gpus=%v warns=%v", gpus, warns)
	}
}

func TestDecodeLEUint(t *testing.T) {
	if v, ok := decodeLEUint([]byte{0, 0, 0, 0, 4, 0, 0, 0}); !ok || v != 16*gib {
		t.Errorf("8 byte: %d %v", v, ok)
	}
	if v, ok := decodeLEUint([]byte{0, 0, 0, 0x40}); !ok || v != gib {
		t.Errorf("4 byte: %d %v", v, ok)
	}
	if _, ok := decodeLEUint([]byte{1, 2, 3}); ok {
		t.Error("3 bytes should fail")
	}
}
