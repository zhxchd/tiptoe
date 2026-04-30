//go:build !unix

package protocol

import "time"

func cpuNow() time.Duration {
	return 0
}
