package fzf

import (
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// FrecencyEntry tracks usage statistics for an item
type FrecencyEntry struct {
	Frequency   uint32 // Number of times this item was selected
	LastAccess  int64  // Unix timestamp of last selection
	FirstAccess int64  // Unix timestamp of first selection
	PrevAccess  int64  // Unix timestamp of previous selection
}

// FrecencyDB manages the frecency database
type FrecencyDB struct {
	entries map[string]*FrecencyEntry
	scores  sync.Map
	path    string
	mutex   sync.RWMutex
	dirty   bool
}

// Time constants for time bucket calculation
const (
	SECOND = 1
	MINUTE = 60 * SECOND
	HOUR   = 60 * MINUTE
	DAY    = 24 * HOUR
	WEEK   = 7 * DAY
)

// Frecency scaling factor for logarithmic transformation to uint16 range
// Calculation: maxUint16 / log₁₀(maxRealisticScore + 1)
//
//	maxRealisticScore ≈ 1,000,000 (extreme power user: 100k freq × 4.0 weight)
//	log₁₀(1,000,001) ≈ 6.0
//	65,535 / 6.0 = 10,922.5
//
// Maps score range [0.25, 1M] → [0, 65535] for direct uint16 use
// Half-life and momentum constants tune the continuous decay and burst boost.
const (
	frecencyScaleFactor = 10922.5
	frecencyHalfLife    = 24 * time.Hour
	momentumWindow      = 6 * time.Hour
	momentumMaxBoost    = 0.5
)

// NewFrecencyDB creates a new frecency database
// If customPath is empty, uses the default platform-specific path
func NewFrecencyDB(customPath string) *FrecencyDB {
	path := customPath
	if path == "" {
		var err error
		path, err = getDefaultFrecencyPath()
		if err != nil {
			// Fallback to current directory if we can't determine default path
			path = "frecency.gob"
		}
	}

	return &FrecencyDB{
		entries: make(map[string]*FrecencyEntry),
		scores:  sync.Map{},
		path:    path,
		dirty:   false,
	}
}

// getDefaultFrecencyPath returns the default platform-specific path for the frecency database
func getDefaultFrecencyPath() (string, error) {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("could not determine user cache directory: %w", err)
	}
	return filepath.Join(cacheDir, "fzf", "frecency.gob"), nil
}

// Load reads the frecency database from disk
// Returns nil if the file doesn't exist (first run)
// Warns to stderr if the file is corrupted and returns empty database
func (db *FrecencyDB) Load() error {
	file, err := os.Open(db.path)
	if err != nil {
		if os.IsNotExist(err) {
			// First run - not an error
			return nil
		}
		return fmt.Errorf("failed to open frecency database: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var entries map[string]*FrecencyEntry
	err = decoder.Decode(&entries)
	if err != nil {
		if err == io.EOF {
			// Empty file - not an error
			return nil
		}
		// Corrupted file - warn but continue with empty database
		fmt.Fprintf(os.Stderr, "Warning: frecency database is corrupted, starting fresh: %v\n", err)
		// Optionally backup the corrupted file
		backupPath := db.path + ".backup"
		if copyErr := copyFile(db.path, backupPath); copyErr == nil {
			fmt.Fprintf(os.Stderr, "Backed up corrupted database to: %s\n", backupPath)
		}
		return nil
	}

	db.mutex.Lock()
	db.entries = entries
	db.dirty = false

	// Calculate all scores after loading entries
	db.scores = sync.Map{}
	for item, entry := range db.entries {
		db.scores.Store(item, db.calculateScore(entry))
	}

	db.mutex.Unlock()
	return nil
}

// Save writes the frecency database to disk atomically
// Uses temp file + rename to ensure atomicity
// Returns nil without doing anything if the database is not dirty
func (db *FrecencyDB) Save() error {
	db.mutex.Lock()
	if !db.dirty {
		db.mutex.Unlock()
		return nil
	}
	db.mutex.Unlock()

	// Create parent directory if it doesn't exist
	dir := filepath.Dir(db.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("failed to create frecency database directory: %w", err)
	}

	tmpFile, err := os.CreateTemp(dir, ".frecency.gob.tmp.*")
	if err != nil {
		return fmt.Errorf("failed to create temporary file: %w", err)
	}
	tmpPath := tmpFile.Name()

	// Ensure cleanup on error
	defer func() {
		if tmpFile != nil {
			tmpFile.Close()
			os.Remove(tmpPath)
		}
	}()

	db.mutex.RLock()
	encoder := gob.NewEncoder(tmpFile)
	err = encoder.Encode(db.entries)
	db.mutex.RUnlock()

	if err != nil {
		return fmt.Errorf("failed to encode frecency database: %w", err)
	}

	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("failed to sync temporary file: %w", err)
	}

	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("failed to close temporary file: %w", err)
	}
	tmpFile = nil // Prevent defer cleanup

	if err := os.Rename(tmpPath, db.path); err != nil {
		return fmt.Errorf("failed to rename temporary file: %w", err)
	}

	// Clear dirty flag after successful save
	db.mutex.Lock()
	db.dirty = false
	db.mutex.Unlock()

	return nil
}

// calculateScore computes the frecency score for a single entry
func (db *FrecencyDB) calculateScore(entry *FrecencyEntry) float64 {
	score, _, _, _ := db.scoreComponents(entry, time.Now())
	return score
}

func (db *FrecencyDB) scoreComponents(entry *FrecencyEntry, now time.Time) (raw, freqComponent, decayComponent, momentumComponent float64) {
	if entry == nil || entry.Frequency == 0 || entry.LastAccess == 0 {
		return 0, 0, 0, 0
	}

	freqComponent = math.Log2(float64(entry.Frequency) + 1.0)

	last := time.Unix(entry.LastAccess, 0)
	duration := now.Sub(last)
	if duration < 0 {
		duration = 0
	}
	halfLifeSeconds := frecencyHalfLife.Seconds()
	if halfLifeSeconds <= 0 {
		halfLifeSeconds = 1
	}
	decayComponent = math.Exp(math.Log(0.5) * duration.Seconds() / halfLifeSeconds)

	momentumComponent = 1.0
	if entry.PrevAccess > 0 {
		prev := time.Unix(entry.PrevAccess, 0)
		delta := last.Sub(prev)
		if delta < 0 {
			delta = 0
		}
		if delta < momentumWindow {
			window := momentumWindow.Seconds()
			if window <= 0 {
				window = 1
			}
			gapRatio := delta.Seconds() / window
			momentumComponent += momentumMaxBoost * (1.0 - gapRatio)
		}
	}

	raw = freqComponent * decayComponent * momentumComponent
	return raw, freqComponent, decayComponent, momentumComponent
}

// GetScore returns the pre-calculated and scaled frecency score for an item
// Returns a value in [0, 65535] range ready for uint16 casting in sort criteria
// Returns 0.0 if the item has never been selected
func (db *FrecencyDB) GetScore(item string) float64 {
	var (
		rawScore float64
		exists   bool
	)

	db.mutex.RLock()
	entry, ok := db.entries[item]
	if ok {
		rawScore, _, _, _ = db.scoreComponents(entry, time.Now())
		exists = true
	}
	db.mutex.RUnlock()

	if exists {
		db.scores.Store(item, rawScore)
	}

	// Apply logarithmic scaling to handle wide dynamic range
	// +1 handles zero/small values, log₁₀ compresses large values
	return math.Min(math.Log10(rawScore+1)*frecencyScaleFactor, 65535)
}

// Update increments the frequency counter and updates the timestamp for an item
// Creates a new entry if the item doesn't exist
func (db *FrecencyDB) Update(item string) {
	now := time.Now().Unix()

	db.mutex.Lock()
	entry, exists := db.entries[item]
	if exists {
		// Update existing entry
		if entry.Frequency < math.MaxUint32 {
			entry.Frequency++
		}
		if entry.LastAccess > 0 {
			entry.PrevAccess = entry.LastAccess
		}
		entry.LastAccess = now
	} else {
		// Create new entry
		entry = &FrecencyEntry{
			Frequency:   1,
			FirstAccess: now,
			LastAccess:  now,
			PrevAccess:  now,
		}
		db.entries[item] = entry
	}
	// Recalculate score for this item
	db.scores.Store(item, db.calculateScore(entry))
	db.dirty = true
	db.mutex.Unlock()
}

// Buff increments the frequency counter for an item
// Creates a new entry if the item doesn't exist
func (db *FrecencyDB) Buff(item string) {
	now := time.Now().Unix()

	db.mutex.Lock()
	defer db.mutex.Unlock()

	entry, exists := db.entries[item]
	if exists {
		// Increment frequency with overflow check
		if entry.Frequency < math.MaxUint32 {
			entry.Frequency++
		}
	} else {
		// Create new entry
		entry = &FrecencyEntry{
			Frequency:   1,
			FirstAccess: now,
			LastAccess:  now,
			PrevAccess:  now,
		}
		db.entries[item] = entry
	}

	// Recalculate score for this item
	db.scores.Store(item, db.calculateScore(entry))
	db.dirty = true
}

// Nerf decrements the frequency counter for an item
// Removes the entry if frequency reaches 0
// Does nothing if the item doesn't exist
func (db *FrecencyDB) Nerf(item string) {
	db.mutex.Lock()
	defer db.mutex.Unlock()

	entry, exists := db.entries[item]
	if !exists {
		return
	}

	// Decrement frequency
	if entry.Frequency > 0 {
		entry.Frequency--
	}

	// Remove entry if frequency reaches 0
	if entry.Frequency == 0 {
		delete(db.entries, item)
		db.scores.Delete(item)
	} else {
		// Recalculate score for this item
		db.scores.Store(item, db.calculateScore(entry))
	}

	db.dirty = true
}

// Remove deletes an entry from the frecency database
// Does nothing if the item doesn't exist
func (db *FrecencyDB) Remove(item string) {
	db.mutex.Lock()
	defer db.mutex.Unlock()

	_, exists := db.entries[item]
	if !exists {
		return
	}

	delete(db.entries, item)
	db.scores.Delete(item)
	db.dirty = true
}

// copyFile copies a file from src to dst
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	return err
}

// frecencyItem holds an item with its score for printing
type frecencyItem struct {
	item       string
	rawScore   float64
	scaled     float64
	frequency  uint32
	lastAccess time.Time
	prevAccess time.Time
	age        time.Duration
	decay      float64
	momentum   float64
	freqScore  float64
}

func formatShortDuration(d time.Duration) string {
	if d < time.Minute {
		return "<1m"
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm", int(d.Minutes()))
	}
	if d < 24*time.Hour {
		return fmt.Sprintf("%dh", int(d.Hours()))
	}
	if d < 7*24*time.Hour {
		return fmt.Sprintf("%dd", int(d.Hours()/(24)))
	}
	if d < 30*24*time.Hour {
		return fmt.Sprintf("%dw", int(d.Hours()/(24*7)))
	}
	if d < 365*24*time.Hour {
		return fmt.Sprintf("%dmo", int(d.Hours()/(24*30)))
	}
	return fmt.Sprintf("%dyr", int(d.Hours()/(24*365)))
}

// printFrecencyTable prints the frecency database and returns exit code
func printFrecencyTable(db *FrecencyDB) (int, error) {
	if db == nil {
		fmt.Println("No frecency data available")
		return ExitOk, nil
	}

	// Print database path
	fmt.Printf("Frecency database: %s\n\n", db.path)

	// Copy entries with scores
	db.mutex.RLock()
	items := make([]frecencyItem, 0, len(db.entries))
	now := time.Now()
	for item, entry := range db.entries {
		rawScore, freqComponent, decayComponent, momentumComponent := db.scoreComponents(entry, now)
		scaled := math.Min(math.Log10(rawScore+1)*frecencyScaleFactor, 65535)
		last := time.Unix(entry.LastAccess, 0)
		age := now.Sub(last)
		if age < 0 {
			age = 0
		}
		var prev time.Time
		if entry.PrevAccess > 0 {
			prev = time.Unix(entry.PrevAccess, 0)
		}

		items = append(items, frecencyItem{
			item:       item,
			rawScore:   rawScore,
			scaled:     scaled,
			frequency:  entry.Frequency,
			lastAccess: last,
			prevAccess: prev,
			age:        age,
			decay:      decayComponent,
			momentum:   momentumComponent,
			freqScore:  freqComponent,
		})
	}
	db.mutex.RUnlock()

	if len(items) == 0 {
		fmt.Println("No frecency data available")
		return ExitOk, nil
	}

	// Sort by score descending
	sort.Slice(items, func(i, j int) bool {
		return items[i].rawScore > items[j].rawScore
	})

	// Print header
	fmt.Printf("%-10s  %-6s  %-16s  %-6s  %-7s  %-9s  %-10s  %s\n", "RAW", "FREQ", "LAST_ACCESS", "AGE", "DECAY", "MOMENTUM", "SCALED", "ITEM")

	// Print entries
	for _, item := range items {
		fmt.Printf("%-10.2f  %-6d  %-16s  %-6s  %-7.3f  %-9.3f  %-10.0f  %s\n",
			item.rawScore,
			item.frequency,
			item.lastAccess.Format("2006-01-02 15:04"),
			formatShortDuration(item.age),
			item.decay,
			item.momentum,
			item.scaled,
			item.item,
		)
	}

	return ExitOk, nil
}
