//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package fzf

import (
	"os"

	"golang.org/x/sys/unix"
)

func lockFrecencyFile(file *os.File) error {
	for {
		err := unix.Flock(int(file.Fd()), unix.LOCK_EX)
		if err != unix.EINTR {
			return err
		}
	}
}
