//go:build windows

package hw

import (
	"fmt"
	"runtime"
	"unsafe"

	"golang.org/x/sys/cpu"
	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

// detectWindowsRAM returns total physical memory via GlobalMemoryStatusEx.
func detectWindowsRAM() (uint64, error) {
	st := memoryStatusEx{Length: uint32(unsafe.Sizeof(memoryStatusEx{}))}
	proc := windows.NewLazySystemDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")
	if r, _, err := proc.Call(uintptr(unsafe.Pointer(&st))); r == 0 {
		return 0, fmt.Errorf("GlobalMemoryStatusEx: %w", err)
	}
	return st.TotalPhys, nil
}

// detectWindowsCPU reads the processor name from the registry. Cores is left
// equal to Threads because NumCPU does not distinguish them.
func detectWindowsCPU() (CPU, error) {
	c := CPU{Threads: runtime.NumCPU(), Cores: runtime.NumCPU(), AVX2: cpu.X86.HasAVX2, AVX512: cpu.X86.HasAVX512F}
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `HARDWARE\DESCRIPTION\System\CentralProcessor\0`, registry.QUERY_VALUE)
	if err != nil {
		return c, fmt.Errorf("open processor key: %w", err)
	}
	defer k.Close()
	name, _, err := k.GetStringValue("ProcessorNameString")
	if err != nil {
		return c, fmt.Errorf("read ProcessorNameString: %w", err)
	}
	c.Model = name
	return c, nil
}
