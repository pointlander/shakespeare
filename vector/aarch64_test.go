// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!noasm && arm) || (!noasm && arm64)
// +build !noasm,arm !noasm,arm64

package vector

import (
	"math/rand"
	"testing"
)

func TestDot(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	x := make([]float32, Size)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := make([]float32, Size)
	for i := range y {
		y[i] = float32(rng.NormFloat64())
	}
	correct := dot(x, y)
	if a := Dot(x, y); int(a*100) != int(correct*100) {
		t.Fatalf("dot product is broken %f != %f", a, correct)
	}
}

func BenchmarkVectorDot(b *testing.B) {
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
		Dot(x, y)
	}
}
