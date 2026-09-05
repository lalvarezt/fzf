package fzf

import (
	"testing"

	"github.com/junegunn/fzf/src/algo"
)

func BenchmarkMatcherHotPath(b *testing.B) {
	original := sortCriteria
	b.Cleanup(func() { sortCriteria = original })
	for _, tc := range []struct {
		name     string
		query    string
		unicode  bool
		frecency bool
		cold     bool
		first    bool
		workers  int
	}{
		{name: "ASCII", query: "file", frecency: true, workers: 1},
		{name: "Unicode", query: "file", unicode: true, frecency: true, workers: 1},
		{name: "ASCIICold", query: "file", frecency: true, cold: true, workers: 1},
		{name: "UnicodeCold", query: "file", unicode: true, frecency: true, cold: true, workers: 1},
		{name: "ASCIIFirst", query: "file", frecency: true, first: true, workers: 1},
		{name: "UnicodeFirst", query: "file", unicode: true, frecency: true, first: true, workers: 1},
		{name: "NoFrecency", query: "file", workers: 1},
		{name: "Sparse", query: "file-001", frecency: true, workers: 1},
		{name: "NoMatch", query: "missing", frecency: true, workers: 1},
		{name: "LongQuery", query: "dir-file-path.txt", frecency: true, workers: 1},
		{name: "Parallel", query: "file", frecency: true, workers: 4},
		{name: "ParallelCold", query: "file", frecency: true, cold: true, workers: 4},
	} {
		b.Run(tc.name, func(b *testing.B) {
			sortCriteria = []criterion{byScore, byLength}
			if tc.frecency {
				sortCriteria = []criterion{byFrecency, byScore, byLength}
			}
			matcher, request, db := prepareFrecencyMatcherBenchmarkData(8000, tc.unicode, tc.workers)
			request.pattern = BuildPattern(matcher.cache, make(map[string]*Pattern), true,
				algo.FuzzyMatchV2, true, CaseSmart, false, true, false, false,
				nil, Delimiter{}, revision{}, []rune(tc.query), nil, 0)
			if tc.frecency {
				request.pattern.frecencyDB = db
			}
			if !tc.first {
				matcher.scan(request)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if tc.cold {
					db.invalidateScoreGeneration()
				}
				if tc.first {
					b.StopTimer()
					db.InvalidateItemScoreCache()
					matcher.cache.Clear()
					clear(matcher.slab)
					clear(matcher.sortBuf)
					b.StartTimer()
				}
				result := matcher.scan(request)
				benchmarkFrecencyMatchCountSink = result.merger.Length()
			}
		})
	}
}
