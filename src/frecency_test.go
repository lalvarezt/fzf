package fzf

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFrecencySelect1(t *testing.T) {
	if path := os.Getenv("FZF_TEST_SELECT1_FRECENCY_FILE"); path != "" {
		opts, err := ParseOptions(false, []string{
			"--frecency", "--select-1", "--ansi", "--with-nth=2", "--accept-nth=1",
			"--frecency-file=" + path,
		})
		if err != nil {
			t.Fatal(err)
		}
		code, err := Run(opts)
		if err != nil {
			t.Fatal(err)
		}
		os.Exit(code)
	}

	path := filepath.Join(t.TempDir(), "frecency.gob")
	for frequency := uint32(1); frequency <= 2; frequency++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestFrecencySelect1$")
		cmd.Env = append(os.Environ(), "FZF_TEST_SELECT1_FRECENCY_FILE="+path)
		cmd.Stdin = strings.NewReader("\x1b[31malpha beta 漢\x1b[0m\n")
		output, err := cmd.CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("auto-selection failed: %v\n%s", err, output)
		}
		if string(output) != "alpha\n" {
			t.Fatalf("unexpected selection output: %q", output)
		}
		db := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, 0.5)
		if err := db.Load(); err != nil {
			t.Fatal(err)
		}
		if entry := db.entries["alpha beta 漢"]; entry == nil || entry.Frequency != frequency {
			t.Fatalf("auto-selection must record the original text %d time(s), got %v", frequency, entry)
		}
		if len(db.entries) != 1 {
			t.Fatalf("expected only the selected item in the database, got %d entries", len(db.entries))
		}
	}
}

func TestFrecencyOptionsRejectNonFiniteValues(t *testing.T) {
	for _, option := range []string{"--frecency-scale-factor", "--frecency-momentum-boost"} {
		for _, value := range []string{"NaN", "Inf", "-Inf"} {
			t.Run(option+"="+value, func(t *testing.T) {
				if _, err := ParseOptions(false, []string{option + "=" + value}); err == nil {
					t.Fatal("expected a finite value to be required")
				}
			})
		}
	}
}

func TestFrecencySaveRenameFailure(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "database")
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	db := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, 0.5)
	db.Update("selected")
	if err := db.Save(); err == nil {
		t.Fatal("expected renaming a file over a directory to fail")
	}
	if !db.dirty {
		t.Fatal("failed save must leave the database dirty")
	}
	matches, err := filepath.Glob(filepath.Join(dir, ".frecency.gob.tmp.*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 0 {
		t.Fatalf("failed save left temporary files: %v", matches)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := db.Save(); err != nil {
		t.Fatal(err)
	}
	reloaded := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, 0.5)
	if err := reloaded.Load(); err != nil {
		t.Fatal(err)
	}
	if entry := reloaded.entries["selected"]; entry == nil || entry.Frequency != 1 {
		t.Fatalf("retry did not persist the selection: %v", entry)
	}
}

func TestFrecencySaveConcurrentUpdate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "frecency.gob")
	db := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, 0.5)
	now := time.Now().Unix()
	for i := range 8000 {
		db.entries[fmt.Sprintf("file-%d", i)] = &FrecencyEntry{Frequency: 1, LastAccess: now, FirstAccess: now}
	}
	for attempt := range 10 {
		db.Update("selected")
		started := make(chan struct{})
		saved := make(chan error, 1)
		go func() {
			close(started)
			saved <- db.Save()
		}()
		<-started
		db.Update("selected")
		if err := <-saved; err != nil {
			t.Fatal(err)
		}
		if err := db.Save(); err != nil {
			t.Fatal(err)
		}
		reloaded := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, 0.5)
		if err := reloaded.Load(); err != nil {
			t.Fatal(err)
		}
		want := uint32(2 * (attempt + 1))
		if entry := reloaded.entries["selected"]; entry == nil || entry.Frequency != want {
			t.Fatalf("concurrent save lost an update: got %v, want frequency %d", entry, want)
		}
	}
}

func TestScoreCacheReplace(t *testing.T) {
	var cache scoreCache
	cache.Replace(map[string]uint16{"first": 1, "second": 2}, 1)
	cache.Store("removed", 5, 1)
	cache.Replace(map[string]uint16{"first": 3, "second": 4}, 2)
	for key, want := range map[string]uint16{"first": 3, "second": 4} {
		if got, ok := cache.Load(key); !ok || got != want {
			t.Errorf("score for %q: got %d, %v; want %d", key, got, ok, want)
		}
		if got, ok := cache.Generation(key); !ok || got != 2 {
			t.Errorf("generation for %q: got %d, %v; want 2", key, got, ok)
		}
	}
	if _, ok := cache.Load("removed"); ok {
		t.Fatal("replacement retained a pending key")
	}
	cache.Store("pending", 5, 2)
	cache.Replace(map[string]uint16{"first": 6, "pending": 7}, 3)
	for key, want := range map[string]uint16{"first": 6, "pending": 7} {
		if got, ok := cache.Load(key); !ok || got != want {
			t.Errorf("score for %q: got %d, %v; want %d", key, got, ok, want)
		}
		if got, ok := cache.Generation(key); !ok || got != 3 {
			t.Errorf("generation for %q: got %d, %v; want 3", key, got, ok)
		}
	}
	if _, ok := cache.Load("second"); ok {
		t.Fatal("replacement retained a removed key")
	}
	cache.Replace(nil, 4)
	if _, ok := cache.Load("first"); ok {
		t.Fatal("empty replacement retained a key")
	}
}
