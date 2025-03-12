// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"
	"testing"
)

func TestCDF(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	cdf := NewCDF16(true)
	filtered := cdf(256)
	for i := 0; i < 1024; i++ {
		filtered.Update(uint16(rng.Intn(256)))
		t.Log(filtered.GetModel())
	}
}
