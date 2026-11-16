//go:build !windows

package main

// enableANSI is a no-op outside Windows, where terminals handle ANSI natively.
func enableANSI() bool { return true }
