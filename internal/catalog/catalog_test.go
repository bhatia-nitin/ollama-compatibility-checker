package catalog

import (
	"strings"
	"testing"
)

func TestEmbeddedValid(t *testing.T) {
	ms := Embedded()
	if err := Validate(ms); err != nil {
		t.Fatalf("Validate(Embedded()) = %v", err)
	}
	if len(ms) < 25 {
		t.Fatalf("len(Embedded()) = %d, want >= 25", len(ms))
	}
	moe := false
	for _, m := range ms {
		if m.ActiveParamsB < m.ParamsB {
			moe = true
		}
	}
	if !moe {
		t.Fatal("no model with ActiveParamsB < ParamsB (MoE) in catalog")
	}
}

func validModel(tag string) Model {
	return Model{
		Family: "fam", Name: "Fam 7B", ParamsB: 7, ActiveParamsB: 7,
		Layers: 32, KVHeads: 8, HeadDim: 128, MaxCtx: 8192, Quality: 50, KVQuantOK: true,
		Variants: []Variant{{Tag: tag, SizeBytes: 4_000_000_000, Bits: 4.57, Quant: "q4_K_M"}},
	}
}

func TestValidateRejects(t *testing.T) {
	zeroLayers := validModel("fam:7b-zero-layers")
	zeroLayers.Layers = 0

	activeOver := validModel("fam:7b-active-over")
	activeOver.ActiveParamsB = 8

	dupA := validModel("fam:7b-dup")
	dupB := validModel("fam:7b-dup")
	dupB.Name = "Fam 7B other"

	cases := []struct {
		name string
		ms   []Model
		tag  string
	}{
		{"zero layers", []Model{zeroLayers}, "fam:7b-zero-layers"},
		{"active > total", []Model{activeOver}, "fam:7b-active-over"},
		{"duplicate tag", []Model{dupA, dupB}, "fam:7b-dup"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := Validate(c.ms)
			if err == nil {
				t.Fatal("Validate returned nil, want error")
			}
			if !strings.Contains(err.Error(), c.tag) {
				t.Fatalf("error %q does not name tag %q", err, c.tag)
			}
		})
	}
	if err := Validate([]Model{validModel("fam:7b-ok")}); err != nil {
		t.Fatalf("valid model rejected: %v", err)
	}
}

func TestQuantPenalty(t *testing.T) {
	cases := map[string]int{
		"q8_0": 0, "q6_K": -1, "q5_K_M": -2, "q4_K_M": -3, "q4_0": -3,
		"q3_K_M": -8, "q2_K": -15, "unknown": 0, "mxfp4": -3,
	}
	for q, want := range cases {
		if got := QuantPenalty(q); got != want {
			t.Errorf("QuantPenalty(%q) = %d, want %d", q, got, want)
		}
	}
}

func TestLoadDefaultsKVQuantOK(t *testing.T) {
	js := `[{"family":"f","name":"F 1B","params_b":1,"active_params_b":1,"layers":16,"kv_heads":8,"head_dim":64,"max_ctx":2048,"quality":30,
	  "variants":[{"tag":"f:1b-q4_K_M","size_bytes":800000000,"bits":6.4,"quant":"q4_K_M"}]},
	 {"family":"g","name":"G 1B","params_b":1,"active_params_b":1,"layers":16,"kv_heads":8,"head_dim":64,"max_ctx":2048,"quality":30,"kv_quant_ok":false,
	  "variants":[{"tag":"g:1b-q4_K_M","size_bytes":800000000,"bits":6.4,"quant":"q4_K_M"}]}]`
	ms, err := Load(strings.NewReader(js))
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) != 2 || !ms[0].KVQuantOK || ms[1].KVQuantOK {
		t.Fatalf("kv_quant_ok defaults wrong: %+v", ms)
	}
	if ms[0].Variants[0].SizeBytes != 800000000 || ms[0].Layers != 16 {
		t.Fatalf("fields not decoded: %+v", ms[0])
	}
}
