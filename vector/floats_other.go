// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

func dot(x, y []float32) (z float32) {
	for i := range x {
		z += x[i] * y[i]
	}
	return z
}
