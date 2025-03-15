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
	"os"
	"runtime"
	"sort"
	"strings"

	"github.com/pointlander/gradient/tf32"

	"github.com/alixaxel/pagerank"
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
	// FlagVectors build vector db
	FlagVectors = flag.Bool("vectors", false, "build vector db")
	// FlagPrompt the prompt to use
	FlagPrompt = flag.String("prompt", "Hello World!", "the prompt to use")
	// FlagCount the number of symbols to generate
	FlagCount = flag.Int("count", 256, "number of symbols to generate")
	// FlagInfer run the model in inference mode
	FlagInfer = flag.String("infer", "", "inference mode")
	// FlagReason reason mode
	FlagReason = flag.String("reason", "", "reason mode")
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

// Infer runs inference on the model
func Infer(symbols map[rune]int, isymbols map[int]rune) {
	rng := rand.New(rand.NewSource(1))

	//m := NewMixer()
	m := NewFiltered()
	for _, v := range []rune(*FlagPrompt) {
		m.Add(byte(symbols[v]))
	}
	set := tf32.NewSet()
	_, _, err := set.Open(*FlagInfer)
	if err != nil {
		panic(err)
	}

	others := tf32.NewSet()
	others.Add("input", 8*256)
	input := others.ByName["input"]
	input.X = input.X[:cap(input.X)]
	l0 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
	l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1")))
	l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
	l3 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
	l4 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w4"), l3), set.Get("b4")))

	path := ""
	cost := float32(0.0)
	for l := 0; l < *FlagCount; l++ {
		q := m.Mix()
		copy(input.X, q[:])
		l4(func(a *tf32.V) bool {
			sum := float32(0.0)
			for _, v := range a.X {
				sum += v
			}
			selection, total := rng.Float32(), float32(0.0)
			for i, v := range a.X {
				total += v / sum
				if selection < total {
					path += fmt.Sprintf("%c", isymbols[i])
					cost += v / sum
					m.Add(byte(i))
					return true
				}
			}
			return true
		})
	}
	fmt.Println(cost)
	fmt.Println(path)
}

// Reason run reason based inference
func Reason(symbols map[rune]int, isymbols map[int]rune) {
	rng := rand.New(rand.NewSource(1))

	//m := NewMixer()
	m := NewFiltered()
	for _, v := range []rune(*FlagPrompt) {
		m.Add(byte(symbols[v]))
	}
	set := tf32.NewSet()
	_, _, err := set.Open(*FlagReason)
	if err != nil {
		panic(err)
	}

	type Vector struct {
		Vector [8 * 256]float32
		Symbol int
	}

	type Vectors struct {
		Vectors []Vector
		Rank    float64
	}

	done := make(chan Vectors, 8)
	process := func(seed int64) {
		others := tf32.NewSet()
		others.Add("input", 8*256)
		input := others.ByName["input"]
		input.X = input.X[:cap(input.X)]
		l0 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
		l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
		l3 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
		l4 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w4"), l3), set.Get("b4")))

		rng := rand.New(rand.NewSource(seed))
		vectors := make([]Vector, *FlagCount)
		cp := m.Copy()
		for j := 0; j < *FlagCount; j++ {
			q := cp.Mix()
			copy(input.X, q[:])
			l4(func(a *tf32.V) bool {
				sum := float32(0.0)
				for _, v := range a.X {
					sum += v
				}
				selection, total := rng.Float32(), float32(0.0)
				for i, v := range a.X {
					total += v / sum
					if selection < total {
						vectors[j] = Vector{
							Vector: q,
							Symbol: i,
						}
						cp.Add(byte(i))
						return true
					}
				}
				return true
			})
		}
		done <- Vectors{
			Vectors: vectors,
		}
	}
	cpus := runtime.NumCPU()
	i, flight := 0, 0
	vectors, index := make([]Vectors, 128), 0
	for i < len(vectors) && flight < cpus {
		go process(rng.Int63())
		flight++
		i++
	}
	for i < len(vectors) {
		vecs := <-done
		vectors[index] = vecs
		index++
		flight--

		go process(rng.Int63())
		flight++
		i++
	}
	for f := 0; f < flight; f++ {
		vecs := <-done
		vectors[index] = vecs
		index++
	}

	graph := pagerank.NewGraph()
	for j := 0; j < len(vectors); j++ {
		for k := 0; k < len(vectors); k++ {
			graph.Link(uint32(j), uint32(k), float64(NCS(vectors[j].Vectors[*FlagCount-1].Vector[:], vectors[k].Vectors[*FlagCount-1].Vector[:])))
		}
	}
	graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
		vectors[node].Rank = rank
	})
	sort.Slice(vectors, func(i, j int) bool {
		return vectors[i].Rank < vectors[j].Rank
	})
	for _, v := range vectors {
		path := ""
		for _, vv := range v.Vectors {
			path += fmt.Sprintf("%c", isymbols[vv.Symbol])
		}
		fmt.Println("--------------------------------------------------------")
		fmt.Println(v.Rank)
		fmt.Println(path)
	}
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
		Infer(symbols, isymbols)
		return
	}

	if *FlagReason != "" {
		Reason(symbols, isymbols)
		return
	}

	if *FlagSmall {
		in = in[:32*1024]
	}

	if *FlagVectors {
		db, err := os.Create("vectors.bin")
		if err != nil {
			panic(err)
		}
		defer db.Close()

		//m := NewMixer()
		m := NewFiltered()
		m.Add(0)
		buffer32 := make([]byte, 4)
		for i, v := range in {
			vector := m.Mix()
			for _, v := range vector {
				bits := math.Float32bits(v)
				for i := range buffer32 {
					buffer32[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := db.Write(buffer32)
				if err != nil {
					panic(err)
				}
				if n != len(buffer32) {
					panic("4 bytes should be been written")
				}
			}
			n, err := db.Write(in[i : i+1])
			if err != nil {
				panic(err)
			}
			if n != 1 {
				panic("1 bytes should be been written")
			}
			m.Add(v)
		}
		return
	}

	type Vector struct {
		Vector [InputSize]float32
		Symbol byte
	}
	vectors, err := os.Open("vectors.bin")
	if err != nil {
		panic(err)
	}
	defer vectors.Close()
	get := func(i int) Vector {
		_, err := vectors.Seek(int64((InputSize*4+1)*i), io.SeekStart)
		if err != nil {
			panic(err)
		}
		vector := Vector{}
		buffer := make([]byte, InputSize*4+1)
		n, err := vectors.Read(buffer)
		if err != nil {
			panic(err)
		}
		if n != len(buffer) {
			panic(fmt.Sprintf("%d bytes should have been read", len(buffer)))
		}
		for k := range vector.Vector {
			var bits uint32
			for l := 0; l < 4; l++ {
				bits |= uint32(buffer[4*k+l]) << (8 * l)
			}
			vector.Vector[k] = math.Float32frombits(bits)
		}
		vector.Symbol = buffer[len(buffer)-1]
		return vector
	}

	others := tf32.NewSet()
	others.Add("input", 8*256, BatchSize)
	others.Add("output", len(symbols), BatchSize)
	input, output := others.ByName["input"], others.ByName["output"]
	input.X = input.X[:cap(input.X)]
	output.X = output.X[:cap(output.X)]
	set := tf32.NewSet()
	set.Add("w0", 8*256, 8*len(symbols))
	set.Add("b0", 8*len(symbols))
	set.Add("w1", 16*len(symbols), 8*len(symbols))
	set.Add("b1", 8*len(symbols))
	set.Add("w2", 16*len(symbols), 8*len(symbols))
	set.Add("b2", 8*len(symbols))
	set.Add("w3", 16*len(symbols), 8*len(symbols))
	set.Add("b3", 8*len(symbols))
	set.Add("w4", 16*len(symbols), len(symbols))
	set.Add("b4", len(symbols))
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
	l0 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
	l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1")))
	l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
	l3 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
	l4 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w4"), l3), set.Get("b4")))
	loss := tf32.Avg(tf32.Quadratic(l4, others.Get("output")))
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
			vector := get(rng.Intn(len(in)))
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
			panic("IsNaN or IsInf")
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
