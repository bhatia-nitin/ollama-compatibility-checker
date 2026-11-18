package catalog

import (
	"strings"
	"testing"
)

func TestFilterTask(t *testing.T) {
	a, b := validModel("a:1b"), validModel("b:1b")
	a.Tasks = []string{"coding"}
	ms := []Model{a, b}
	if got := FilterTask(ms, ""); len(got) != 2 {
		t.Fatalf("empty task: got %d models, want 2", len(got))
	}
	if got := FilterTask(ms, "coding"); len(got) != 1 || got[0].Variants[0].Tag != "a:1b" {
		t.Fatalf("coding: got %+v", got)
	}
	if got := FilterTask(ms, "vision"); len(got) != 0 {
		t.Fatalf("vision: got %d models, want 0", len(got))
	}
}

func TestValidateRejectsUnknownTask(t *testing.T) {
	m := validModel("a:1b")
	m.Tasks = []string{"chat"}
	if err := Validate([]Model{m}); err == nil || !strings.Contains(err.Error(), `unknown task "chat"`) {
		t.Fatalf("got %v", err)
	}
}

// Every task must have at least one model in the embedded catalog, or --task
// would always come back empty.
func TestEmbeddedCoversEveryTask(t *testing.T) {
	for _, task := range Tasks {
		if len(FilterTask(Embedded(), task)) == 0 {
			t.Errorf("no embedded model tagged %q", task)
		}
	}
}
