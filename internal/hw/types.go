// Package hw holds hardware description types and memory bandwidth tables.
package hw

// GPU describes one detected accelerator.
type GPU struct {
	Vendor                 string // "nvidia" | "amd" | "apple" | "intel"
	Name                   string
	VRAMBytes              uint64
	FreeVRAMBytes          uint64
	BandwidthGBs           float64
	BandwidthLowConfidence bool
	Backend                string // "cuda" | "rocm" | "metal" | "vulkan" | "cpu"
	Supported              bool
	Driver                 string
	ComputeCap             string // NVIDIA "8.9"; AMD gfx target "gfx1100"
}

// CPU describes the host processor.
type CPU struct {
	Model   string
	Cores   int
	Threads int
	AVX2    bool
	AVX512  bool
}

// Hardware is the full detected machine profile.
type Hardware struct {
	OS              string
	Arch            string
	GPUs            []GPU
	RAMBytes        uint64
	RAMBandwidthGBs float64
	CPU             CPU
	UnifiedMemory   bool // Apple Silicon
	Warnings        []string
}
