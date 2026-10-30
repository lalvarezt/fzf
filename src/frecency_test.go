package fzf

import (
	"bufio"
	"bytes"
	"context"
	"encoding/gob"
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
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gob.NewEncoder(file).Encode(db.entries); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
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
		if len(reloaded.entries) != 8001 {
			t.Fatalf("save lost unrelated history: %d entries", len(reloaded.entries))
		}
	}
}

func TestScoreCacheReplace(t *testing.T) {
	var cache scoreCache
	cache.Replace(map[string]uint16{"first": 1, "second": 2}, 1)
	for i, tc := range []struct {
		name, pending string
		values        map[string]uint16
	}{
		{"same keys", "removed", map[string]uint16{"first": 3, "second": 4}},
		{"different keys", "pending", map[string]uint16{"first": 6, "pending": 7}},
		{"empty", "", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			generation := uint32(i + 2)
			if tc.pending != "" {
				cache.Store(tc.pending, 5, generation-1)
			}
			cache.Replace(tc.values, generation)
			for _, key := range []string{"first", "second", "pending", "removed"} {
				want, exists := tc.values[key]
				if got, ok := cache.Load(key); ok != exists || got != want {
					t.Errorf("score for %q: got %d, %v; want %d, %v", key, got, ok, want, exists)
				}
				if got, ok := cache.Generation(key); ok != exists || exists && got != generation {
					t.Errorf("generation for %q: got %d, %v; want %d, %v", key, got, ok, generation, exists)
				}
			}
		})
	}
}

func TestFrecencyOverlappingSessions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.gob")
	first := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, .5)
	second := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, .5)
	for _, db := range []*FrecencyDB{first, second} {
		if err := db.Load(); err != nil {
			t.Fatal(err)
		}
		db.Update("shared")
	}
	first.Update("first")
	second.Update("second")
	for _, db := range []*FrecencyDB{first, second} {
		if err := db.Save(); err != nil {
			t.Fatal(err)
		}
	}
	reloaded := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, .5)
	if err := reloaded.Load(); err != nil {
		t.Fatal(err)
	}
	for key, frequency := range map[string]uint32{"shared": 2, "first": 1, "second": 1} {
		if entry := reloaded.entries[key]; entry == nil || entry.Frequency != frequency {
			t.Errorf("overlapping sessions lost %q: got %v, want frequency %d", key, entry, frequency)
		}
	}
}

func TestFrecencyConcurrentProcesses(t *testing.T) {
	if path := os.Getenv("FZF_TEST_CONCURRENT_FRECENCY_FILE"); path != "" {
		db := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, .5)
		if err := db.Load(); err != nil {
			t.Fatal(err)
		}
		fmt.Println("ready")
		if _, err := bufio.NewReader(os.Stdin).ReadString('\n'); err != nil {
			t.Fatal(err)
		}
		for range 5 {
			db.Update("shared")
			db.Update(os.Getenv("FZF_TEST_CONCURRENT_FRECENCY_ITEM"))
			if err := db.Save(); err != nil {
				t.Fatal(err)
			}
			if err := db.Save(); err != nil {
				t.Fatal(err)
			}
		}
		return
	}
	path := filepath.Join(t.TempDir(), "history.gob")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	var commands []*exec.Cmd
	var errors []*bytes.Buffer
	var release []func()
	for i := range 4 {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestFrecencyConcurrentProcesses$")
		cmd.Env = append(os.Environ(), "FZF_TEST_CONCURRENT_FRECENCY_FILE="+path, fmt.Sprintf("FZF_TEST_CONCURRENT_FRECENCY_ITEM=worker-%d", i))
		stderr := &bytes.Buffer{}
		cmd.Stderr = stderr
		input, err := cmd.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		output, err := cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		commands = append(commands, cmd)
		errors = append(errors, stderr)
		if line, err := bufio.NewReader(output).ReadString('\n'); err != nil || line != "ready\n" {
			t.Fatalf("worker did not load history: %q, %v", line, err)
		}
		release = append(release, func() { fmt.Fprintln(input, "save"); input.Close() })
	}
	for _, start := range release {
		start()
	}
	for i, cmd := range commands {
		if err := cmd.Wait(); err != nil {
			t.Fatalf("worker %d: %v\n%s", i, err, errors[i])
		}
	}
	db := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, .5)
	if err := db.Load(); err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]uint32{"shared": 20, "worker-0": 5, "worker-1": 5, "worker-2": 5, "worker-3": 5} {
		if entry := db.entries[key]; entry == nil || entry.Frequency != want {
			t.Errorf("lost %q: got %v, want %d", key, entry, want)
		}
	}
}

func TestFrecencyOverlappingAdjustments(t *testing.T) {
	for _, action := range []string{"nerf", "remove", "select", "unrelated after removal"} {
		t.Run(action, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "history.gob")
			first := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, .5)
			first.Buff("item")
			first.Buff("item")
			if err := first.Save(); err != nil {
				t.Fatal(err)
			}
			second := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, .5)
			if err := second.Load(); err != nil {
				t.Fatal(err)
			}
			if action == "unrelated after removal" {
				first.Remove("item")
			} else {
				first.applyOperation(frecencyOperation{"item", frecencySelect, 1800000200})
			}
			var want *FrecencyEntry
			switch action {
			case "nerf":
				second.Nerf("item")
				want = &FrecencyEntry{Frequency: 2}
			case "remove":
				second.Remove("item")
			case "select":
				second.applyOperation(frecencyOperation{"item", frecencySelect, 1800000100})
				want = &FrecencyEntry{Frequency: 4}
			case "unrelated after removal":
				second.Update("other")
			}
			if err := first.Save(); err != nil {
				t.Fatal(err)
			}
			if err := second.Save(); err != nil {
				t.Fatal(err)
			}
			if err := second.Save(); err != nil {
				t.Fatal(err)
			}
			if err := first.Load(); err != nil {
				t.Fatal(err)
			}
			got := first.entries["item"]
			if want == nil {
				if got != nil {
					t.Fatalf("removed item resurrected: %v", got)
				}
			} else if got == nil || got.Frequency != want.Frequency || got.LastAccess != 1800000200 {
				t.Fatalf("wrong merged history: %v", got)
			} else if action == "select" && got.PrevAccess != 1800000100 {
				t.Fatalf("timestamps follow save order instead of selection order: %v", got)
			}
			if first.GetScore("item") != second.GetScore("item") {
				t.Fatal("save left stale scores")
			}
		})
	}
}

func TestFrecencyLoadRecovery(t *testing.T) {
	for _, state := range []string{"missing", "empty", "corrupt", "nil map"} {
		t.Run(state, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "history.gob")
			if state != "missing" {
				var data bytes.Buffer
				if state == "corrupt" {
					data.WriteString("invalid history")
				}
				if state == "nil map" {
					if err := gob.NewEncoder(&data).Encode(map[string]*FrecencyEntry(nil)); err != nil {
						t.Fatal(err)
					}
				}
				if err := os.WriteFile(path, data.Bytes(), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			db := NewFrecencyDB(path, 3150, 30*24*time.Hour, 6*time.Hour, .5)
			if err := db.Load(); err != nil {
				t.Fatal(err)
			}
			db.Update("selected")
			if err := db.Save(); err != nil {
				t.Fatal(err)
			}
			if err := db.Load(); err != nil {
				t.Fatal(err)
			}
			if entry := db.entries["selected"]; len(db.entries) != 1 || entry == nil || entry.Frequency != 1 {
				t.Fatalf("recovery lost selection: %v", db.entries)
			}
			if state == "corrupt" {
				backup, err := os.ReadFile(path + ".backup")
				if err != nil || string(backup) != "invalid history" {
					t.Fatalf("corrupt history not backed up: %q, %v", backup, err)
				}
			}
		})
	}
}
