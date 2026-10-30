package fzf

import (
	"fmt"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/junegunn/fzf/src/util"
)

var benchmarkFrecencyScoreSink uint16

func prepareFrecencyScoreItem(text string) (*FrecencyDB, *Item) {
	db := NewFrecencyDB("", 3150.0, 30*24*time.Hour, 6*time.Hour, 0.5)
	now := db.sessionTime.Unix()
	entry := &FrecencyEntry{
		Frequency:   64,
		FirstAccess: now - int64(90*24*3600),
		LastAccess:  now - int64(3600),
		PrevAccess:  now - int64(7200),
	}
	db.entries[text] = entry
	db.calculateAndStoreScore(text, entry)
	return db, &Item{text: util.ToChars([]byte(text))}
}

func TestFrecencyItemLifecycle(t *testing.T) {
	for _, mode := range []string{"ASCII", "Unicode", "transformed"} {
		for _, indexed := range []bool{false, true} {
			// Plain ASCII bypasses both per-item caches.
			if mode == "ASCII" && indexed {
				continue
			}
			t.Run(fmt.Sprintf("%s/indexed=%v", mode, indexed), func(t *testing.T) {
				key := "file-a"
				if mode != "ASCII" {
					key = "file-漢"
				}
				item := &Item{text: util.ToChars([]byte(key))}
				if mode == "transformed" {
					original := []byte("\x1b[31m" + key + "\x1b[0m")
					item.origText = &original
					item.text = util.ToChars([]byte("display"))
				}
				db := NewFrecencyDB(filepath.Join(t.TempDir(), "frecency.gob"), 3150, 30*24*time.Hour, 6*time.Hour, .5)
				db.stripAnsi = true
				if indexed {
					db.prepareItemScoreTable(0, 1)
				}
				if got := db.GetScoreForItem(item); got != 0 {
					t.Fatalf("unseen item scored %d", got)
				}
				for _, step := range []struct {
					name      string
					act       func(*Item)
					frequency uint32
				}{
					{"nerf unseen", db.NerfItem, 0},
					{"remove unseen", db.RemoveItem, 0},
					{"select", db.UpdateItem, 1},
					{"repeat", db.UpdateItem, 2},
					{"buff", db.BuffItem, 3},
					{"nerf", db.NerfItem, 2},
					{"remove", db.RemoveItem, 0},
					{"buff unseen", db.BuffItem, 1},
					{"nerf last use", db.NerfItem, 0},
					{"select again", db.UpdateItem, 1},
				} {
					t.Run(step.name, func(t *testing.T) {
						var before FrecencyEntry
						if entry := db.entries[key]; entry != nil {
							before = *entry
						}
						step.act(item)
						entry := db.entries[key]
						if step.frequency == 0 {
							if entry != nil || len(db.entries) != 0 {
								t.Fatalf("removed item retained: %v", db.entries)
							}
						} else {
							if entry == nil || entry.Frequency != step.frequency || len(db.entries) != 1 {
								t.Fatalf("wrong history: %v", db.entries)
							}
							if before.Frequency != 0 {
								if entry.FirstAccess != before.FirstAccess {
									t.Fatal("first selection timestamp changed")
								}
								if step.name == "buff" || step.name == "nerf" {
									if entry.LastAccess != before.LastAccess || entry.PrevAccess != before.PrevAccess {
										t.Fatal("manual adjustment changed recency")
									}
								} else if entry.PrevAccess != before.LastAccess {
									t.Fatal("repeat did not remember previous selection")
								}
							}
						}
						for range 2 {
							got := db.GetScoreForItem(item)
							if got != db.GetScore(key) || (got == 0) != (step.frequency == 0) {
								t.Fatalf("stale score: %d, stored %d", got, db.GetScore(key))
							}
						}
						if err := db.Save(); err != nil {
							t.Fatal(err)
						}
						reloaded := NewFrecencyDB(db.path, 3150, 30*24*time.Hour, 6*time.Hour, .5)
						if err := reloaded.Load(); err != nil {
							t.Fatal(err)
						}
						if !reflect.DeepEqual(reloaded.entries, db.entries) {
							t.Fatalf("history changed after save/load: %v", reloaded.entries)
						}
					})
				}
			})
		}
	}
}

func TestGetScoreForItemInvalidationRefreshesTextKey(t *testing.T) {
	db, item := prepareFrecencyScoreItem("first-漢")
	db.prepareItemScoreTable(item.Index(), item.Index()+1)
	if db.GetScoreForItem(item) == 0 {
		t.Fatal("expected initial score")
	}

	item.text = util.ToChars([]byte("second-漢"))
	db.Update("second-漢")
	db.InvalidateItemScoreCache()
	if got, want := db.GetScoreForItem(item), db.GetScore("second-漢"); got != want {
		t.Fatalf("expected score %d after text invalidation, got %d", want, got)
	}
}

func TestGetScoreForItemIndexedItemReplacement(t *testing.T) {
	db, item := prepareFrecencyScoreItem("first-漢")
	db.prepareItemScoreTable(item.Index(), item.Index()+1)
	if db.GetScoreForItem(item) == 0 {
		t.Fatal("expected initial score")
	}

	replacement := &Item{text: util.ToChars([]byte("replacement-漢"))}
	replacement.text.Index = item.Index()
	db.Update("replacement-漢")
	if got, want := db.GetScoreForItem(replacement), db.GetScore("replacement-漢"); got != want {
		t.Fatalf("expected replacement score %d, got %d", want, got)
	}
}

func TestGetScoreForItemConcurrentInvalidation(t *testing.T) {
	db, item := prepareFrecencyScoreItem("concurrent-漢")
	db.prepareItemScoreTable(item.Index(), item.Index()+1)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 1000; j++ {
				if j%10 == 0 {
					db.invalidateScoreGeneration()
				}
				db.GetScoreForItem(item)
			}
		}()
	}
	wg.Wait()
}

func benchmarkGetScoreForItem(b *testing.B, text string, warm bool) {
	db, item := prepareFrecencyScoreItem(text)
	if warm {
		benchmarkFrecencyScoreSink = db.GetScoreForItem(item)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !warm {
			db.invalidateScoreGeneration()
		}
		benchmarkFrecencyScoreSink = db.GetScoreForItem(item)
	}
}

func BenchmarkGetScoreForItemWarmASCII(b *testing.B) {
	benchmarkGetScoreForItem(b, "file-001.txt", true)
}

func BenchmarkGetScoreForItemWarmUnicode(b *testing.B) {
	benchmarkGetScoreForItem(b, "file-\u6f22\u5b57-\u03b4.txt", true)
}

func BenchmarkGetScoreForItemColdASCII(b *testing.B) {
	benchmarkGetScoreForItem(b, "file-001.txt", false)
}

func BenchmarkGetScoreForItemColdUnicode(b *testing.B) {
	benchmarkGetScoreForItem(b, "file-\u6f22\u5b57-\u03b4.txt", false)
}
