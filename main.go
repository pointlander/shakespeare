// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/pointlander/gradient/tf32"
)

//go:embed books/*
var Data embed.FS

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-4
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

const (
	// InputSize is the size of the input
	InputSize = Size * 256
	// BatchSize is the size of a batch
	BatchSize = 100
)

func main() {
	file, err := Data.Open("books/100.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	data, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}
	in, symbols, isymbols, s := make([]byte, 0, 8), make(map[rune]int), make(map[int]rune), 0
	for _, v := range string(data) {
		if _, has := symbols[v]; !has {
			symbols[v] = s
			isymbols[s] = v
			s++
		}
		in = append(in, byte(symbols[v]))
	}
	fmt.Println(s)

	in = in[:32*1024]
	type Vector struct {
		Vector [InputSize]float32
		Symbol byte
	}
	mind, index := make([]Vector, len(in)), 0
	m := NewMixer()
	m.Add(0)

	others := tf32.NewSet()
	others.Add("input", 8*256, BatchSize)
	others.Add("output", len(symbols), BatchSize)
	input, output := others.ByName["input"], others.ByName["output"]
	input.X = input.X[:cap(input.X)]
	output.X = output.X[:cap(output.X)]
	for _, v := range in {
		index = (index + 1) % len(mind)
		mind[index].Vector = m.Mix()
		mind[index].Symbol = v
		m.Add(v)
	}
	rng := rand.New(rand.NewSource(1))
	set := tf32.NewSet()
	set.Add("w", 8*256, 4*len(symbols))
	set.Add("b", 4*len(symbols))
	set.Add("w1", 4*len(symbols), len(symbols))
	set.Add("b1", len(symbols))
	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}
	l := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w"), others.Get("input")), set.Get("b")))
	l1 := tf32.Add(tf32.Mul(set.Get("w1"), l), set.Get("b1"))
	loss := tf32.Avg(tf32.Quadratic(l1, others.Get("output")))
	for i := 0; i < 1024; i++ {
		pow := func(x float32) float32 {
			y := math.Pow(float64(x), float64(i+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return float32(y)
		}

		for j := 0; j < BatchSize; j++ {
			vector := mind[rng.Intn(len(mind))]
			copy(input.X[j*8*256:(j+1)*8*256], vector.Vector[:])
			for k := 0; k < len(symbols); k++ {
				output.X[j*len(symbols)+k] = 0
			}
			output.X[j*len(symbols)+int(vector.Symbol)] = 1
		}

		others.Zero()
		set.Zero()

		cost := tf32.Gradient(loss).X[0]
		if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
			break
		}

		norm := float32(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = float32(math.Sqrt(float64(norm)))
		b1, b2 := pow(B1), pow(B2)
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}
		fmt.Println(i, cost)
	}

	{
		m := NewMixer()
		for _, v := range []rune("Hello world!") {
			m.Add(byte(symbols[v]))
		}
		others := tf32.NewSet()
		others.Add("input", 8*256, 1)
		input := others.ByName["input"]
		input.X = input.X[:cap(input.X)]
		l = tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w"), others.Get("input")), set.Get("b")))
		l1 = tf32.Add(tf32.Mul(set.Get("w1"), l), set.Get("b1"))
		type Path struct {
			Path string
			Cost float32
		}
		paths := make([]Path, 0, 8)
		for j := 0; j < 128; j++ {
			query := ""
			cost := float32(0.0)
			i := 0
			cp := m.Copy()
			for {
				q := cp.Mix()
				copy(input.X, q[:])
				max, symbol := float32(0.0), 0
				l1(func(a *tf32.V) bool {
					sum := float32(0.0)
					for _, v := range a.X {
						sum += v
					}
					selection, total := rng.Float32(), float32(0.0)
					for i, v := range a.X {
						total += v / sum
						if selection < total {
							cost += v / sum
							symbol = i
							break
						}
						/*if v > max {
							max, symbol = v, is[i]
						}*/
					}
					return true
				})
				_ = max
				query += fmt.Sprintf("%c", isymbols[symbol])
				cp.Add(byte(symbol))
				i++
				if i >= 128 && (symbol == '.' || symbol == '!' || symbol == '?') {
					break
				}
				if i >= 1024 {
					break
				}
			}
			paths = append(paths, Path{
				Path: query,
				Cost: cost,
			})
		}
		for i := range paths {
			paths[i].Cost /= float32(len(paths[i].Path))
		}
		sort.Slice(paths, func(i, j int) bool {
			return paths[i].Cost > paths[j].Cost
		})
		fmt.Printf(paths[0].Path)
	}
}
