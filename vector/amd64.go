// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && amd64
// +build !noasm,amd64

package vector

import (
	"unsafe"
)

func Dot(x, y []float32) (z float32) {
	_mm256_dot(unsafe.Pointer(&x[0]), unsafe.Pointer(&y[0]), unsafe.Pointer(uintptr(len(x))), unsafe.Pointer(&z))
	return z
}
