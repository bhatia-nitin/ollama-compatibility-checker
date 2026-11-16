package hw

import (
	"bufio"
	"context"
	"fmt"
	"strconv"
	"strings"
)

// parseMemTotal reads MemTotal (in kB) from /proc/meminfo content and
// returns bytes.
func parseMemTotal(data []byte) (uint64, error) {
	sc := bufio.NewScanner(strings.NewReader(string(data)))
	for sc.Scan() {
		rest, ok := strings.CutPrefix(sc.Text(), "MemTotal:")
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		if len(f) == 0 {
			break
		}
		kb, err := strconv.ParseUint(f[0], 10, 64)
		if err != nil {
			return 0, fmt.Errorf("MemTotal: %w", err)
		}
		return kb * 1024, nil
	}
	return 0, fmt.Errorf("MemTotal not found")
}

// sysctlUint runs `sysctl -n key` and parses the output as an unsigned int.
func sysctlUint(ctx context.Context, run Runner, key string) (uint64, error) {
	out, err := run(ctx, "sysctl", "-n", key)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
}
