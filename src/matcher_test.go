package fzf

import (
	"slices"
	"testing"
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
