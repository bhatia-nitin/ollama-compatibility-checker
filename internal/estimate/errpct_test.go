package estimate

import "testing"

func TestErrPct(t *testing.T) {
	for _, c := range []struct {
		backend string
		frac    float64
		low     bool
		want    int
	}{
		{"metal", 1, false, 15},
		{"cuda", 1, false, 30},
		{"", 0, false, 30},
		{"metal", 0.5, false, 40},
		{"cuda", 1, true, 50},
	} {
		if got := errPct(c.backend, c.frac, c.low); got != c.want {
			t.Errorf("errPct(%q, %v, %v) = %d, want %d", c.backend, c.frac, c.low, got, c.want)
		}
	}
	lo, hi := Estimate{TokS: 20, ErrPct: 15}.TokSRange()
	if lo != 17 || hi != 23 {
		t.Errorf("range = %v..%v, want 17..23", lo, hi)
	}
}
