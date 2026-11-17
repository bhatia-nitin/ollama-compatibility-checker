package main

import (
	"os"
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/calib"
)

// TestMain pins the fitted speed factors, so these tests do not change each
// time a calibration record is added to calibration/data.
func TestMain(m *testing.M) {
	calib.UseFactors(calib.Table{"metal": {Eff: 0.568, ErrPct: 15, Runs: 1, Machines: 1}})
	os.Exit(m.Run())
}
