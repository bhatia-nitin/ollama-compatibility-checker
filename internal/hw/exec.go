package hw

import (
	"context"
	"os/exec"
	"time"
)

// Runner runs an external command and returns its combined output and error.
// Detectors take a Runner so tests can inject canned output.
type Runner func(ctx context.Context, name string, args ...string) ([]byte, error)

// commandTimeout bounds every command run through DefaultRunner.
const commandTimeout = 5 * time.Second

// DefaultRunner runs the command with a 5 s timeout layered on ctx.
var DefaultRunner Runner = func(ctx context.Context, name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, commandTimeout)
	defer cancel()
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
}
