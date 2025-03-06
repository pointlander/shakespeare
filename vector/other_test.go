// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import (
	"math/rand"
	"testing"
)

const Size = 32 * 1024

func BenchmarkDot(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, Size)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := make([]float32, Size)
	for i := range y {
		y[i] = float32(rng.NormFloat64())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dot(x, y)
	}
}
