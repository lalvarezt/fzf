//go:build aix || solaris

package fzf

import (
	"os"

	"golang.org/x/sys/unix"
)

func lockFrecencyFile(file *os.File) error {
	lock := unix.Flock_t{Type: unix.F_WRLCK, Whence: 0, Len: 1}
	for {
		err := unix.FcntlFlock(file.Fd(), unix.F_SETLKW, &lock)
		if err != unix.EINTR {
			return err
		}
	}
}
