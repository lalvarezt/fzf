package algo

import (
	"crypto/sha256"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/junegunn/fzf/src/util"
)

func TestFuzzyMatchV2ScoreRows(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	slab := util.MakeSlab(100000, 10000)
	smallSlab := util.MakeSlab(10000, 10000)
	for i := range slab.I16 {
		slab.I16[i] = 12345
	}
	for i := range smallSlab.I16 {
		smallSlab.I16[i] = 12345
	}
	digest := sha256.New()
	for sample := range 1500 {
		alphabet := []rune("abcABCéÉ漢字 /_-123")
		if sample%3 == 0 {
			alphabet = []rune("abcABC /_-123")
		}
		text := make([]rune, 8+rng.Intn(280))
		for i := range text {
			text[i] = alphabet[rng.Intn(len(alphabet))]
		}
		pattern := make([]rune, 3+rng.Intn(min(20, len(text)-2)))
		for i := range pattern {
			if sample%2 == 0 {
				pattern[i] = text[i*len(text)/len(pattern)]
			} else {
				pattern[i] = alphabet[rng.Intn(len(alphabet))]
			}
		}
		input := util.ToChars([]byte(string(text)))
		for _, sensitive := range []bool{false, true} {
			for _, normalize := range []bool{false, true} {
				query := append([]rune(nil), pattern...)
				if !sensitive {
					query = []rune(strings.ToLower(string(query)))
				}
				if normalize {
					for i, r := range query {
						query[i] = normalizeRune(r)
					}
				}
				for _, forward := range []bool{false, true} {
					full, _ := FuzzyMatchV2(sensitive, normalize, forward, &input, query, true, nil)
					plain, _ := FuzzyMatchV2(sensitive, normalize, forward, &input, query, false, nil)
					got, pos := FuzzyMatchV2(sensitive, normalize, forward, &input, query, false, slab)
					if got != plain || got.Score != full.Score || got.End != full.End || pos != nil {
						t.Fatalf("sample %d sensitive=%v normalize=%v forward=%v: full=%v plain=%v reused=%v",
							sample, sensitive, normalize, forward, full, plain, got)
					}
					small, _ := FuzzyMatchV2(sensitive, normalize, forward, &input, query, false, smallSlab)
					if small != plain {
						t.Fatalf("sample %d: small slab returned %v, expected %v", sample, small, plain)
					}
					fmt.Fprintln(digest, got)
				}
			}
		}
	}
	// The digest also permits comparison with the pre-optimization executable,
	// including Start, which differs between scoring and backtracking.
	t.Logf("score-only results: %x", digest.Sum(nil))
}

func TestFuzzyMatchV2ScoreRowsSlabCapacity(t *testing.T) {
	input := util.ToChars([]byte("abc abc"))
	pattern := []rune("abc")
	want, _ := FuzzyMatchV2(true, false, true, &input, pattern, true, nil)
	// The initial rows and bonuses take 21 entries; full matrices take 42.
	// alloc16 requires capacity strictly greater than the requested end offset.
	for _, size := range []int{62, 63, 64} {
		slab := util.MakeSlab(size, 32)
		var got Result
		allocs := testing.AllocsPerRun(100, func() {
			got, _ = FuzzyMatchV2(true, false, true, &input, pattern, false, slab)
		})
		if allocs != 0 || got.Score != want.Score || got.End != want.End {
			t.Fatalf("slab capacity %d: got %v with %g allocations, expected score/end from %v without allocations", size, got, allocs, want)
		}
	}
}

var benchmarkHotPathResult Result

func BenchmarkFuzzyMatchV2ScoreRows(b *testing.B) {
	for _, tc := range []struct {
		name    string
		input   string
		pattern string
	}{
		{"Path", "dir-000/file-00000-path.txt", "dir-file-path.txt"},
		{"Short", strings.Repeat("a_bc/def ghij-klmn/opqr_stuv/wxyz ", 4), "abcdefghijklmnopqrstuvwx"},
		{"Long", strings.Repeat("a_bc/def ghij-klmn/opqr_stuv/wxyz ", 64), "abcdefghijklmnopqrstuvwx"},
	} {
		input := util.ToChars([]byte(tc.input))
		pattern := []rune(tc.pattern)
		for _, withSlab := range []bool{false, true} {
			b.Run(fmt.Sprintf("%s/Slab=%v", tc.name, withSlab), func(b *testing.B) {
				var slab *util.Slab
				if withSlab {
					slab = util.MakeSlab(100000, 10000)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					benchmarkHotPathResult, _ = FuzzyMatchV2(false, false, true, &input, pattern, false, slab)
				}
			})
		}
	}
}
