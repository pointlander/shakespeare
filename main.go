// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"flag"
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

var (
	// FlagInfer run the model in inference mode
	FlagInfer = flag.String("infer", "", "inference mode")
	/// FlagSmall is small mode
	FlagSmall = flag.Bool("small", false, "small mode")
)

// Quadratic computes the quadratic cost of two tensors
func Quadratic(k tf32.Continuation, node int, a, b *tf32.V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, size := tf32.NewV(a.S[1]), len(a.X)
	stddevs := []float32{}
	targets := []int{}
	for i := 0; i < size; i += width {
		av, bv, sum := a.X[i:i+width], b.X[i:i+width], float32(0.0)
		index := 0
		for j, bx := range bv {
			if bx == 1 {
				index = j
				break
			}
		}
		avg := float32(0.0)
		for _, ax := range av {
			avg += ax
		}
		avg /= float32(len(av))
		stddev := float32(0.0)
		for _, ax := range av {
			diff := ax - avg
			stddev += diff * diff
		}
		stddev = float32(math.Sqrt(float64(stddev) / float64(len(av))))
		target := av[index]
		targets = append(targets, index)
		stddevs = append(stddevs, stddev)
		for j, ax := range av {
			p := (ax - (target - stddev))
			if j == index {
				p = (ax - target)
			}
			sum += p * p
		}
		c.X = append(c.X, .5*sum)
	}
	if k(&c) {
		return true
	}
	index := 0
	for i := 0; i < size; i += width {
		av, ad, bd, d := a.X[i:i+width], a.D[i:i+width], b.D[i:i+width], c.D[index]
		target := av[targets[index]]
		stddev := stddevs[index]
		for j, ax := range av {
			b := target - stddev
			if j == targets[index] {
				b = target
			}
			ad[j] += (ax - b) * d
			bd[j] += (b - ax) * d
		}
		index++
	}
	return false
}

func main() {
	flag.Parse()

	rng := rand.New(rand.NewSource(1))

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

	if *FlagInfer != "" {
		m := NewMixer()
		for _, v := range []rune("Hello world!") {
			m.Add(byte(symbols[v]))
		}
		set := tf32.NewSet()
		_, _, err := set.Open(*FlagInfer)
		if err != nil {
			panic(err)
		}
		others := tf32.NewSet()
		others.Add("input", 8*256, 1)
		input := others.ByName["input"]
		input.X = input.X[:cap(input.X)]
		l0 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
		l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1")))
		l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
		l3 := tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3"))
		type Path struct {
			Path string
			Cost float32
		}
		paths := make([]Path, 0, 8)
		for j := 0; j < 256; j++ {
			query := ""
			cost := float32(0.0)
			i := 0
			cp := m.Copy()
			for {
				q := cp.Mix()
				copy(input.X, q[:])
				max, symbol := float32(0.0), 0
				l3(func(a *tf32.V) bool {
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
					}
					/*cp := make([]float32, len(a.X))
					copy(cp, a.X)
					softmax(cp)
					selection, total := rng.Float32(), float32(0.0)
					for i, v := range cp {
						total += v
						if selection < total {
							cost += v
							symbol = i
							break
						}
					}*/
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
		return
	}

	if *FlagSmall {
		in = in[:32*1024]
	} else {
		in = in[:2*1024*1024]
	}
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
	set := tf32.NewSet()
	set.Add("w0", 8*256, 8*len(symbols))
	set.Add("b0", 8*len(symbols))
	set.Add("w1", 8*len(symbols), 8*len(symbols))
	set.Add("b1", 8*len(symbols))
	set.Add("w2", 8*len(symbols), 8*len(symbols))
	set.Add("b2", 8*len(symbols))
	set.Add("w3", 8*len(symbols), len(symbols))
	set.Add("b3", len(symbols))
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
	//quadratic := tf32.B(Quadratic)
	l0 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
	l1 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1")))
	l2 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
	l3 := tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3"))
	loss := tf32.Avg(tf32.Quadratic(l3, others.Get("output")))
	iterations := 3 * 60 * 1024
	if *FlagSmall {
		iterations = 1024
	}
	cost := float32(0.0)
	for i := 0; i < iterations; i++ {
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

		cost = tf32.Gradient(loss).X[0]
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
	set.Save("set.bin", cost, iterations)
}
