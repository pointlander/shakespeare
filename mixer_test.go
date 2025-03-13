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
	for i := 1; i < 9; i++ {
		cdf := NewCDF16(true)
		filtered := cdf(256, i)
		for j := 0; j < 1024; j++ {
			filtered.Update(uint16(rng.Intn(256)))
			t.Log(filtered.GetModel())
		}
	}
}

func TestCDFCopy(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := 1; i < 9; i++ {
		cdf := NewCDF16(true)
		filtered := cdf(256, i)
		cp := filtered.Copy()
		for j := 0; j < 1024; j++ {
			x := rng.Intn(256)
			filtered.Update(uint16(x))
			cp.Update(uint16(x))
			t.Log(filtered.GetModel())
		}
		a, b := filtered.GetModel(), cp.GetModel()
		for i, v := range a {
			if v != b[i] {
				t.Fatalf("%d != %d", v, b[i])
			}
		}
	}
}

func TestFiltered(t *testing.T) {
	a := NewFiltered()
	a.Add(1)
	a.Add(1)
	b := NewFiltered()
	b.Add(1)
	c := NewFiltered()
	c.Add(1)
	x := a.Mix()
	y := b.Mix()
	z := c.Mix()
	i := NCS(z[:], x[:])
	j := NCS(z[:], y[:])
	if j < i {
		t.Fatalf("%f < %f", j, i)
	}
}
