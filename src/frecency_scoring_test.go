package fzf

import (
	"math"
	"testing"
	"time"
)

func TestFrecencyScoring(t *testing.T) {
	now := time.Unix(1800000000, 0)
	for _, tc := range []struct {
		name      string
		frequency uint32
		age, gap  time.Duration
		previous  bool
		want      float64
	}{
		{name: "unselected"},
		{name: "first selection", frequency: 1, want: 1},
		{name: "frequency grows logarithmically", frequency: 7, want: 3},
		{name: "one half life", frequency: 7, age: 30 * 24 * time.Hour, want: 1.5},
		{name: "two half lives", frequency: 7, age: 60 * 24 * time.Hour, want: .75},
		{name: "immediate repeat", frequency: 3, previous: true, want: 3},
		{name: "half momentum window", frequency: 3, previous: true, gap: 3 * time.Hour, want: 2.5},
		{name: "momentum boundary", frequency: 3, previous: true, gap: 6 * time.Hour, want: 2},
		{name: "outside momentum window", frequency: 3, previous: true, gap: 7 * time.Hour, want: 2},
		{name: "future timestamp", frequency: 1, age: -time.Hour, want: 1},
		{name: "reversed timestamps", frequency: 3, previous: true, gap: -time.Hour, want: 3},
		{name: "very old selection", frequency: 1, age: 3650 * 24 * time.Hour, want: math.Exp2(-3650.0 / 30)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			db := NewFrecencyDB("", 3150, 30*24*time.Hour, 6*time.Hour, .5)
			db.sessionTime = now
			entry := &FrecencyEntry{Frequency: tc.frequency, LastAccess: now.Add(-tc.age).Unix()}
			if tc.previous {
				entry.PrevAccess = entry.LastAccess - int64(tc.gap/time.Second)
			}
			raw, _, _, _ := db.scoreComponents(entry, now)
			if math.Abs(raw-tc.want) > 1e-12 {
				t.Fatalf("raw score: got %g, want %g", raw, tc.want)
			}
			db.calculateAndStoreScore("item", entry)
			if got, want := db.GetScore("item"), uint16(tc.want*3150); got != want {
				t.Fatalf("scaled score: got %d, want %d", got, want)
			}
		})
	}
}

func TestFrecencyScoringConfiguration(t *testing.T) {
	now := time.Unix(1800000000, 0)
	for _, tc := range []struct {
		name             string
		scale, boost     float64
		halfLife, window time.Duration
		want             uint16
	}{
		{"custom half life and momentum", 1000, 1, 24 * time.Hour, 2 * time.Hour, 1500},
		{"disabled momentum", 1000, 0, 24 * time.Hour, 2 * time.Hour, 1000},
		{"expired momentum", 1000, 1, 24 * time.Hour, time.Hour, 1000},
		{"saturated score", 100000, 1, 24 * time.Hour, 2 * time.Hour, math.MaxUint16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			db := NewFrecencyDB("", tc.scale, tc.halfLife, tc.window, tc.boost)
			db.sessionTime = now
			last := now.Add(-24 * time.Hour)
			db.calculateAndStoreScore("item", &FrecencyEntry{Frequency: 3, LastAccess: last.Unix(), PrevAccess: last.Add(-time.Hour).Unix()})
			if got := db.GetScore("item"); got != tc.want {
				t.Fatalf("got %d, want %d", got, tc.want)
			}
		})
	}
}

func TestFrecencyManualAdjustmentsPreserveHistory(t *testing.T) {
	for _, tc := range []struct {
		name            string
		frequency, want uint32
		buff            bool
	}{
		{"buff", 2, 3, true},
		{"nerf", 2, 1, false},
		{"saturated frequency", math.MaxUint32, math.MaxUint32, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			db := NewFrecencyDB("", 3150, 30*24*time.Hour, 6*time.Hour, .5)
			before := FrecencyEntry{Frequency: tc.frequency, FirstAccess: 1700000000, PrevAccess: 1700003600, LastAccess: 1700007200}
			entry := before
			db.entries["item"] = &entry
			if tc.buff {
				db.Buff("item")
			} else {
				db.Nerf("item")
			}
			want := before
			want.Frequency = tc.want
			if entry != want {
				t.Fatalf("got %+v, want %+v", entry, want)
			}
		})
	}
}
