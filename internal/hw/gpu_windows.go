//go:build windows

package hw

import (
	"fmt"

	"golang.org/x/sys/windows/registry"
)

const displayClassKey = `SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}`

// readRegValue reads a numeric value stored as REG_QWORD, REG_DWORD, or
// little endian REG_BINARY (4 or 8 bytes).
func readRegValue(k registry.Key, name string) (uint64, bool) {
	_, typ, err := k.GetValue(name, nil)
	if err != nil {
		return 0, false
	}
	switch typ {
	case registry.QWORD, registry.DWORD:
		v, _, err := k.GetIntegerValue(name)
		return v, err == nil
	case registry.BINARY:
		b, _, err := k.GetBinaryValue(name)
		if err != nil {
			return 0, false
		}
		return decodeLEUint(b)
	}
	return 0, false
}

// readRegMemory reads adapter memory: qwMemorySize, falling back to MemorySize.
func readRegMemory(k registry.Key) uint64 {
	for _, name := range []string{"HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"} {
		if v, ok := readRegValue(k, name); ok {
			return v
		}
	}
	return 0
}

// DetectWindowsRegistry enumerates numeric subkeys of the display class key.
// Subkeys that cannot be opened are skipped silently.
func DetectWindowsRegistry() ([]RegAdapter, error) {
	root, err := registry.OpenKey(registry.LOCAL_MACHINE, displayClassKey, registry.ENUMERATE_SUB_KEYS)
	if err != nil {
		return nil, fmt.Errorf("open display class key: %w", err)
	}
	defer root.Close()
	names, err := root.ReadSubKeyNames(-1)
	if err != nil {
		return nil, fmt.Errorf("list display adapters: %w", err)
	}
	var out []RegAdapter
	for _, sub := range names {
		if len(sub) != 4 || !isDigits(sub) {
			continue
		}
		k, err := registry.OpenKey(registry.LOCAL_MACHINE, displayClassKey+`\`+sub, registry.QUERY_VALUE)
		if err != nil {
			continue
		}
		a := RegAdapter{QwMemorySize: readRegMemory(k)}
		a.DriverDesc, _, _ = k.GetStringValue("DriverDesc")
		a.DriverVersion, _, _ = k.GetStringValue("DriverVersion")
		k.Close()
		out = append(out, a)
	}
	return out, nil
}

func isDigits(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}
