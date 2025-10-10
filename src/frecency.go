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
}

// FrecencyDB manages the frecency database
type FrecencyDB struct {
	entries map[string]*FrecencyEntry
	scores  map[string]float64
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
//   maxRealisticScore ≈ 1,000,000 (extreme power user: 100k freq × 4.0 weight)
//   log₁₀(1,000,001) ≈ 6.0
//   65,535 / 6.0 = 10,922.5
// Maps score range [0.25, 1M] → [0, 65535] for direct uint16 use
const frecencyScaleFactor =  10922.5

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
		scores:  make(map[string]float64),
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
	db.scores = make(map[string]float64, len(db.entries))
	for item, entry := range db.entries {
		db.scores[item] = db.calculateScore(entry)
	}

	db.mutex.Unlock()
	return nil
}

// Save writes the frecency database to disk atomically
// Uses temp file + rename to ensure atomicity
func (db *FrecencyDB) Save() error {
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
	frequency := float64(entry.Frequency)
	lastAccess := entry.LastAccess
	duration := max(time.Now().Unix()-lastAccess, 0)

	// Time buckets (zoxide-like algorithm)
	var timeWeight float64
	if duration < HOUR {
		timeWeight = 4.0
	} else if duration < DAY {
		timeWeight = 2.0
	} else if duration < WEEK {
		timeWeight = 0.5
	} else {
		timeWeight = 0.25
	}

	return frequency * timeWeight
}

// GetScore returns the pre-calculated and scaled frecency score for an item
// Returns a value in [0, 65535] range ready for uint16 casting in sort criteria
// Returns 0.0 if the item has never been selected
func (db *FrecencyDB) GetScore(item string) float64 {
	db.mutex.RLock()
	rawScore := db.scores[item]
	db.mutex.RUnlock()

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
		entry.LastAccess = now
	} else {
		// Create new entry
		entry = &FrecencyEntry{
			Frequency:   1,
			FirstAccess: now,
			LastAccess:  now,
		}
		db.entries[item] = entry
	}
	// Recalculate score for this item
	db.scores[item] = db.calculateScore(entry)
	db.dirty = true
	db.mutex.Unlock()
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
	score      float64
	frequency  uint32
	lastAccess time.Time
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
	for item, entry := range db.entries {
		items = append(items, frecencyItem{
			item:       item,
			score:      db.scores[item],
			frequency:  entry.Frequency,
			lastAccess: time.Unix(entry.LastAccess, 0),
		})
	}
	db.mutex.RUnlock()

	if len(items) == 0 {
		fmt.Println("No frecency data available")
		return ExitOk, nil
	}

	// Sort by score descending
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})

	// Print header
	fmt.Printf("%-10s  %-10s  %-20s  %s\n", "SCORE", "FREQUENCY", "LAST_ACCESS", "ITEM")

	// Print entries
	for _, item := range items {
		fmt.Printf("%-10.1f  %-10d  %-20s  %s\n",
			item.score,
			item.frequency,
			item.lastAccess.Format("2006-01-02T15:04:05"),
			item.item,
		)
	}

	return ExitOk, nil
}
