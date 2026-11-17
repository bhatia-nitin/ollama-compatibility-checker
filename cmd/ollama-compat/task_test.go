package main

import (
	"strings"
	"testing"
)

func TestTaskFlag(t *testing.T) {
	for _, args := range [][]string{{"--task", "chat"}, {"recommend", "--task", "chat"}, {"bench", "--task", "chat"}} {
		_, _, err := run(t, args...)
		if err == nil || !strings.Contains(err.Error(), "coding, reasoning, vision") {
			t.Errorf("%v: want error listing valid tasks, got %v", args, err)
		}
	}
	out, _, err := run(t, "--task", "coding")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "task: coding") || !strings.Contains(out, "qwen2.5-coder") || strings.Contains(out, "gemma") {
		t.Errorf("coding output:\n%s", out)
	}
}
