package fzf

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/junegunn/fzf/src/algo"
	"github.com/junegunn/fzf/src/util"
)

var benchmarkFrecencyResultSink Result
var benchmarkFrecencyMatchCountSink int
var benchmarkFrecencyRawScoreSink float64

func makeFrecencyBenchmarkEntry(now int64, index, frequencyMod, firstAccessDays int) *FrecencyEntry {
	last := now - int64(index%86400)
	return &FrecencyEntry{
		Frequency:   uint32(index%frequencyMod + 1),
		FirstAccess: now - int64(firstAccessDays*24*3600),
		LastAccess:  last,
		PrevAccess:  last - int64(index%3600),
	}
}

func prepareFrecencyBenchmarkData(itemCount int, unicode bool) (*FrecencyDB, []*Item) {
	db := NewFrecencyDB("", 3150.0, 30*24*time.Hour, 6*time.Hour, 0.5)
	now := db.sessionTime.Unix()
	items := make([]*Item, itemCount)

	for i := range itemCount {
		var text string
		if unicode {
			text = fmt.Sprintf("archivo-%05d-漢字-δ", i)
		} else {
			text = fmt.Sprintf("file-%05d-path.txt", i)
		}

		chars := util.ToChars([]byte(text))
		items[i] = &Item{text: chars}

		entry := makeFrecencyBenchmarkEntry(now, i, 512, 90)
		db.entries[text] = entry
	}
	db.rebuildScores()

	return db, items
}

func BenchmarkFrecencyScoreComponents(b *testing.B) {
	db, _ := prepareFrecencyBenchmarkData(8000, true)
	entries := make([]*FrecencyEntry, 0, len(db.entries))
	for _, entry := range db.entries {
		entries = append(entries, entry)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, _, _, _ := db.scoreComponents(entries[i%len(entries)], db.sessionTime)
		benchmarkFrecencyRawScoreSink = raw
	}
}

func BenchmarkFrecencyScoreRebuild(b *testing.B) {
	db, _ := prepareFrecencyBenchmarkData(8000, true)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db.rebuildScores()
	}
}

func BenchmarkFrecencyScoreRebuildMap(b *testing.B) {
	db, _ := prepareFrecencyBenchmarkData(8000, true)
	scores := make(map[string]uint16, len(db.entries))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clear(scores)
		for item, entry := range db.entries {
			raw, _, _, _ := db.scoreComponents(entry, db.sessionTime)
			scores[item] = uint16(min(raw*db.scaleFactor, 65535))
		}
	}
}

func benchmarkBuildResultFrecency(b *testing.B, unicode bool) {
	original := sortCriteria
	sortCriteria = []criterion{byFrecency}
	b.Cleanup(func() {
		sortCriteria = original
	})

	const itemCount = 4096
	db, items := prepareFrecencyBenchmarkData(itemCount, unicode)
	mask := itemCount - 1

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkFrecencyResultSink = buildResult(items[i&mask], nil, 123, db)
	}
}

func BenchmarkBuildResultFrecencyASCII(b *testing.B) {
	benchmarkBuildResultFrecency(b, false)
}

func BenchmarkBuildResultFrecencyUnicode(b *testing.B) {
	benchmarkBuildResultFrecency(b, true)
}

func prepareFrecencyMatcherBenchmarkData(itemCount int, unicode bool, threads ...int) (*Matcher, MatchRequest, *FrecencyDB) {
	cache := NewChunkCache()
	db := NewFrecencyDB("", 3150.0, 30*24*time.Hour, 6*time.Hour, 0.5)
	now := db.sessionTime.Unix()

	itemIndex := int32(0)
	chunkList := NewChunkList(cache, func(item *Item, data []byte) bool {
		item.text = util.ToChars(data)
		item.text.Index = itemIndex
		itemIndex++
		return true
	})

	for i := range itemCount {
		var text string
		if unicode {
			text = fmt.Sprintf("dir-%03d/archivo-%05d-漢字-file.txt", i%128, i)
		} else {
			text = fmt.Sprintf("dir-%03d/file-%05d-path.txt", i%128, i)
		}

		entry := makeFrecencyBenchmarkEntry(now, i, 1024, 180)
		db.entries[text] = entry

		chunkList.Push([]byte(text))
	}
	db.rebuildScores()

	chunks, _, _ := chunkList.Snapshot(0)

	pattern := BuildPattern(
		cache,
		make(map[string]*Pattern),
		true,
		algo.FuzzyMatchV2,
		true,
		CaseSmart,
		false,
		true,
		false,
		false,
		nil,
		Delimiter{},
		revision{},
		[]rune("file"),
		nil,
		0,
	)
	pattern.frecencyDB = db

	workerCount := 1
	if len(threads) > 0 {
		workerCount = threads[0]
	}
	matcher := NewMatcher(cache, nil, true, false, util.NewEventBox(), revision{}, workerCount)
	request := MatchRequest{
		chunks:   chunks,
		pattern:  pattern,
		sort:     true,
		revision: revision{},
	}

	return matcher, request, db
}

func benchmarkMatcherScanFrecency(b *testing.B, unicode bool, cold bool) {
	original := sortCriteria
	sortCriteria = []criterion{byFrecency, byScore, byLength}
	b.Cleanup(func() {
		sortCriteria = original
	})

	const itemCount = 8000
	matcher, request, db := prepareFrecencyMatcherBenchmarkData(itemCount, unicode)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if cold {
			db.invalidateScoreGeneration()
		}
		result := matcher.scan(request)
		benchmarkFrecencyMatchCountSink = result.merger.Length()
	}
}

func BenchmarkMatcherScanFrecencyASCII(b *testing.B) {
	benchmarkMatcherScanFrecency(b, false, false)
}

func BenchmarkMatcherScanFrecencyUnicode(b *testing.B) {
	benchmarkMatcherScanFrecency(b, true, false)
}

func BenchmarkMatcherScanFrecencyASCIICold(b *testing.B) {
	benchmarkMatcherScanFrecency(b, false, true)
}

func BenchmarkMatcherScanFrecencyUnicodeCold(b *testing.B) {
	benchmarkMatcherScanFrecency(b, true, true)
}

func benchmarkMatcherScanFrecencyParallel(b *testing.B, unicode bool, cold bool) {
	original := sortCriteria
	sortCriteria = []criterion{byFrecency, byScore, byLength}
	b.Cleanup(func() {
		sortCriteria = original
	})

	const itemCount = 8000
	matcher, request, db := prepareFrecencyMatcherBenchmarkData(itemCount, unicode, runtime.GOMAXPROCS(0))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if cold {
			db.invalidateScoreGeneration()
		}
		result := matcher.scan(request)
		benchmarkFrecencyMatchCountSink = result.merger.Length()
	}
}

func BenchmarkMatcherScanFrecencyASCIIParallel(b *testing.B) {
	benchmarkMatcherScanFrecencyParallel(b, false, false)
}

func BenchmarkMatcherScanFrecencyUnicodeParallel(b *testing.B) {
	benchmarkMatcherScanFrecencyParallel(b, true, false)
}

func BenchmarkMatcherScanFrecencyASCIIColdParallel(b *testing.B) {
	benchmarkMatcherScanFrecencyParallel(b, false, true)
}

func BenchmarkMatcherScanFrecencyUnicodeColdParallel(b *testing.B) {
	benchmarkMatcherScanFrecencyParallel(b, true, true)
}
