package fzf

import (
	"slices"
	"testing"
	"time"

	"github.com/junegunn/fzf/src/util"
)

func TestScanPreservesPreviousResults(t *testing.T) {
	original := sortCriteria
	sortCriteria = []criterion{byScore, byLength}
	t.Cleanup(func() { sortCriteria = original })
	for _, workers := range []int{1, 4} {
		matcher, request, _ := prepareFrecencyMatcherBenchmarkData(2500, false, workers)
		first := matcher.scan(request).merger
		if first.Length() != 2500 {
			t.Fatalf("workers=%d: expected 2500 matches, got %d", workers, first.Length())
		}
		want := make([][]Result, len(first.lists))
		for i, list := range first.lists {
			want[i] = slices.Clone(list)
		}
		request.pattern = buildPatternWith(matcher.cache, []rune("file-001"))
		second := matcher.scan(request).merger
		if second.Length() == 0 || second.Length() >= first.Length() {
			t.Fatalf("workers=%d: expected a smaller nonempty result set", workers)
		}
		for i, list := range want {
			if !slices.Equal(first.lists[i], list) {
				t.Fatalf("workers=%d: second scan changed previous result list %d", workers, i)
			}
		}
	}
}

func TestFrecencyRankingAfterUsageChanges(t *testing.T) {
	original := sortCriteria
	sortCriteria = []criterion{byFrecency, byScore, byLength}
	t.Cleanup(func() { sortCriteria = original })
	cache := NewChunkCache()
	db := NewFrecencyDB("", 3150, 30*24*time.Hour, 6*time.Hour, 0)
	var index int32
	list := NewChunkList(cache, func(item *Item, data []byte) bool {
		item.text = util.ToChars(data)
		item.text.Index = index
		index++
		return true
	})
	for _, text := range []string{"file-a", "file-b", "other"} {
		list.Push([]byte(text))
	}
	chunks, _, _ := list.Snapshot(0)
	db.Update("other") // Usage must not promote an item that does not match.
	events := util.NewEventBox()
	matcher := NewMatcher(cache, func(query []rune) *Pattern {
		p := buildPatternWith(cache, query)
		p.frecencyDB = db
		return p
	}, true, false, events, revision{}, 1)
	done := make(chan struct{})
	go func() { matcher.Loop(); close(done) }()
	t.Cleanup(func() { matcher.Stop(); <-done })
	search := func(query string, want []string) {
		t.Helper()
		matcher.Reset(chunks, []rune(query), false, true, true, revision{})
		result := make(chan *Merger, 1)
		go func() {
			for {
				var merger *Merger
				events.Wait(func(es *util.Events) {
					if value, ok := (*es)[EvtSearchFin]; ok {
						merger = value.(MatchResult).merger
					}
					es.Clear()
				})
				if merger != nil {
					result <- merger
					return
				}
			}
		}()
		select {
		case merger := <-result:
			got := make([]string, merger.Length())
			for i := range got {
				got[i] = merger.Get(i).item.AsString(false)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("query %q: got %v, want %v", query, got, want)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("matcher did not finish")
		}
	}
	search("file", []string{"file-a", "file-b"})
	search("file", []string{"file-a", "file-b"}) // Warm the merger cache.
	db.Buff("file-b")
	search("file", []string{"file-b", "file-a"})
	search("file-b", []string{"file-b"})
	db.Remove("file-b")
	search("file", []string{"file-a", "file-b"})
	db.Update("file-b")
	search("file", []string{"file-b", "file-a"})
	db.Nerf("file-b")
	search("file", []string{"file-a", "file-b"})
}
