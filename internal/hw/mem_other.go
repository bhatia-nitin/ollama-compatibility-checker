//go:build !windows

package hw

import "errors"

var errNotWindows = errors.New("not supported on this OS")

func detectWindowsRAM() (uint64, error) { return 0, errNotWindows }

func detectWindowsCPU() (CPU, error) { return CPU{}, errNotWindows }

// DetectWindowsRegistry is a stub so Detect compiles on every OS.
func DetectWindowsRegistry() ([]RegAdapter, error) { return nil, errNotWindows }
