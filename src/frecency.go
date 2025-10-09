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
	path    string
	mutex   sync.RWMutex
	dirty   bool
}

// Time constants for time bucket calculation
const (
	secondsPerHour  = 60 * 60
	secondsPerDay   = 24 * secondsPerHour
	secondsPerWeek  = 7 * secondsPerDay
	secondsPerMonth = 30 * secondsPerDay
)

// Time weight multipliers for different age buckets
const (
	weightVeryRecent = 4.0  // < 1 hour
	weightRecent     = 2.0  // < 1 day
	weightModerate   = 1.0  // < 1 week
	weightOld        = 0.5  // < 1 month
	weightVeryOld    = 0.25 // older than 1 month
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

// GetScore calculates the frecency score for an item
// Returns 0.0 if the item has never been selected
func (db *FrecencyDB) GetScore(item string) float64 {
	db.mutex.RLock()
	entry, exists := db.entries[item]
	if !exists {
		db.mutex.RUnlock()
		return 0.0
	}
	frequency := entry.Frequency
	lastAccess := entry.LastAccess
	db.mutex.RUnlock()

	timeWeight := getTimeWeight(lastAccess)
	return float64(frequency) * timeWeight
}

// Update increments the frequency counter and updates the timestamp for an item
// Creates a new entry if the item doesn't exist
func (db *FrecencyDB) Update(item string) {
	now := time.Now().Unix()

	db.mutex.Lock()
	entry, exists := db.entries[item]
	if exists {
		// Update existing entry
		// Use saturating addition to prevent overflow
		if entry.Frequency < math.MaxUint32 {
			entry.Frequency++
		}
		entry.LastAccess = now
	} else {
		// Create new entry
		db.entries[item] = &FrecencyEntry{
			Frequency:   1,
			FirstAccess: now,
			LastAccess:  now,
		}
	}
	db.dirty = true
	db.mutex.Unlock()
}

// calculateScore computes the frecency score using frequency and time weight
func calculateScore(entry *FrecencyEntry) float64 {
	timeWeight := getTimeWeight(entry.LastAccess)
	return float64(entry.Frequency) * timeWeight
}

// getTimeWeight returns a time-based multiplier based on how recently the item was accessed
func getTimeWeight(lastAccess int64) float64 {
	now := time.Now().Unix()
	ageSeconds := now - lastAccess

	// Handle clock skew (future timestamps)
	ageSeconds = max(ageSeconds, 0)

	switch {
	case ageSeconds < secondsPerHour:
		return weightVeryRecent
	case ageSeconds < secondsPerDay:
		return weightRecent
	case ageSeconds < secondsPerWeek:
		return weightModerate
	case ageSeconds < secondsPerMonth:
		return weightOld
	default:
		return weightVeryOld
	}
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

	// Copy entries with scores while holding lock
	db.mutex.RLock()
	items := make([]frecencyItem, 0, len(db.entries))
	for item, entry := range db.entries {
		items = append(items, frecencyItem{
			item:       item,
			score:      calculateScore(entry),
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
