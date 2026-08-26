package fzf

import (
	"sync"
	"sync/atomic"
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

func TestGetScoreForItemInvalidation(t *testing.T) {
	db, item := prepareFrecencyScoreItem("file-漢-001.txt")

	first := db.GetScoreForItem(item)
	if first == 0 {
		t.Fatal("expected non-zero score")
	}

	again := db.GetScoreForItem(item)
	if again != first {
		t.Fatalf("expected cached score %d, got %d", first, again)
	}
	cached, ok := db.itemScores.Load(item)
	if !ok {
		t.Fatal("expected cached item score")
	}
	firstGen := atomic.LoadUint32(&cached.(*cachedItemScore).generation)

	db.Buff("file-漢-001.txt")
	updated := db.GetScoreForItem(item)
	if updated != db.GetScore("file-漢-001.txt") {
		t.Fatalf("expected updated score %d, got %d", db.GetScore("file-漢-001.txt"), updated)
	}
	cached, ok = db.itemScores.Load(item)
	if !ok {
		t.Fatal("expected updated cached item score")
	}
	if atomic.LoadUint32(&cached.(*cachedItemScore).generation) == firstGen {
		t.Fatal("expected generation bump after DB update")
	}
}

func TestGetScoreForItemUsesOriginalText(t *testing.T) {
	db := NewFrecencyDB("", 3150.0, 30*24*time.Hour, 6*time.Hour, 0.5)
	original := []byte("alpha beta gamma")
	item := &Item{
		text:     util.ToChars([]byte("beta")),
		origText: &original,
	}

	db.UpdateItem(item)

	if db.GetScore("alpha beta gamma") == 0 {
		t.Fatal("expected score to be stored under original item text")
	}
	if db.GetScore("beta") != 0 {
		t.Fatal("did not expect score to be stored under transformed text")
	}
	if got := db.GetScoreForItem(item); got != db.GetScore("alpha beta gamma") {
		t.Fatalf("expected item score %d, got %d", db.GetScore("alpha beta gamma"), got)
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
