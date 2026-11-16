package hw

import (
	"bufio"
	"context"
	"strconv"
	"strings"
)

// ParseCPUInfo parses /proc/cpuinfo content. Threads is the number of
// "processor" entries; Cores is "cpu cores" times the number of distinct
// physical ids (at least one package).
func ParseCPUInfo(data []byte) CPU {
	var c CPU
	packages := map[string]bool{}
	perPackage := 0
	sc := bufio.NewScanner(strings.NewReader(string(data)))
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		switch k {
		case "processor":
			c.Threads++
		case "model name":
			if c.Model == "" {
				c.Model = v
			}
		case "physical id":
			packages[v] = true
		case "cpu cores":
			if n, err := strconv.Atoi(v); err == nil {
				perPackage = n
			}
		case "flags":
			for _, f := range strings.Fields(v) {
				switch f {
				case "avx2":
					c.AVX2 = true
				case "avx512f":
					c.AVX512 = true
				}
			}
		}
	}
	pk := len(packages)
	if pk == 0 {
		pk = 1
	}
	c.Cores = perPackage * pk
	if c.Cores == 0 {
		c.Cores = c.Threads
	}
	return c
}

// detectDarwinCPU reads CPU details through sysctl. Failures leave fields
// zero; the leaf7 feature flags exist only on Intel Macs.
func detectDarwinCPU(ctx context.Context, run Runner) CPU {
	var c CPU
	if out, err := run(ctx, "sysctl", "-n", "machdep.cpu.brand_string"); err == nil {
		c.Model = strings.TrimSpace(string(out))
	}
	if n, err := sysctlUint(ctx, run, "hw.physicalcpu"); err == nil {
		c.Cores = int(n)
	}
	if n, err := sysctlUint(ctx, run, "hw.logicalcpu"); err == nil {
		c.Threads = int(n)
	}
	if out, err := run(ctx, "sysctl", "-n", "machdep.cpu.leaf7_features"); err == nil {
		for _, f := range strings.Fields(strings.ToUpper(string(out))) {
			switch f {
			case "AVX2":
				c.AVX2 = true
			case "AVX512F":
				c.AVX512 = true
			}
		}
	}
	return c
}
