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
	"github.com/pointlander/shakespeare/vector"

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
	InputSize = (Size + Order + 1) * 256
	// BatchSize is the size of a batch
	BatchSize = 100
)

var (
	// FlagVDB is the vector database mode
	FlagVDB = flag.Bool("vdb", false, "vector database mode")
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
	// FlagMixer is the mixer type
	FlagMixer = flag.String("mixer", "filtered", "mixer type")
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

	var m Mix
	if *FlagMixer == "filtered" {
		m = NewFiltered()
	} else {
		m = NewMixer()
	}
	for _, v := range []rune(*FlagPrompt) {
		m.Add(byte(symbols[v]))
	}
	set := tf32.NewSet()
	_, _, err := set.Open(*FlagInfer)
	if err != nil {
		panic(err)
	}

	others := tf32.NewSet()
	others.Add("input", InputSize)
	input := others.ByName["input"]
	input.X = input.X[:cap(input.X)]
	l0 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
	l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1")))
	l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
	l3 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
	l4 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w4"), l3), set.Get("b4")))
	l5 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w5"), l4), set.Get("b5")))
	l6 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w6"), l5), set.Get("b6")))
	l7 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w7"), l6), set.Get("b7")))

	path := ""
	cost := float32(0.0)
	for l := 0; l < *FlagCount; l++ {
		q := m.Mix()
		copy(input.X, q[:])
		l7(func(a *tf32.V) bool {
			aa := a.X[:len(symbols)]
			sum := float32(0.0)
			for _, v := range aa {
				sum += v
			}
			selection, total := rng.Float32(), float32(0.0)
			for i, v := range aa {
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

	var m Mix
	if *FlagMixer == "filtered" {
		m = NewFiltered()
	} else {
		m = NewMixer()
	}
	for _, v := range []rune(*FlagPrompt) {
		m.Add(byte(symbols[v]))
	}
	set := tf32.NewSet()
	_, _, err := set.Open(*FlagReason)
	if err != nil {
		panic(err)
	}

	type Vector struct {
		Vector [InputSize]float32
		Symbol int
	}

	type Vectors struct {
		Vectors []Vector
		Rank    float64
	}

	done := make(chan Vectors, 8)
	process := func(seed int64) {
		others := tf32.NewSet()
		others.Add("input", InputSize)
		input := others.ByName["input"]
		input.X = input.X[:cap(input.X)]
		l0 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
		l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1")))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
		l3 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
		l4 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w4"), l3), set.Get("b4")))
		l5 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w5"), l4), set.Get("b5")))
		l6 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w6"), l5), set.Get("b6")))
		l7 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w7"), l6), set.Get("b7")))

		rng := rand.New(rand.NewSource(seed))
		vectors := make([]Vector, *FlagCount)
		cp := m.Copy()
		for j := 0; j < *FlagCount; j++ {
			q := cp.Mix()
			copy(input.X, q[:])
			l7(func(a *tf32.V) bool {
				aa := a.X[:len(symbols)]
				sum := float32(0.0)
				for _, v := range aa {
					sum += v
				}
				selection, total := rng.Float32(), float32(0.0)
				for i, v := range aa {
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
		return vectors[i].Rank > vectors[j].Rank
	})

	for i := 0; i < 33; i++ {
		for j := range vectors {
			for k := 0; k < 8; k++ {
				vector := Vectors{
					Vectors: make([]Vector, *FlagCount),
				}
				copy(vector.Vectors, vectors[j].Vectors)
				var m Mix
				if *FlagMixer == "filtered" {
					m = NewFiltered()
				} else {
					m = NewMixer()
				}
				m.Add(0)
				n := rng.Intn(*FlagCount)
				for l := 0; l < n; l++ {
					m.Add(byte(vectors[j].Vectors[l].Symbol))
				}

				others := tf32.NewSet()
				others.Add("input", InputSize)
				input := others.ByName["input"]
				input.X = input.X[:cap(input.X)]
				l0 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
				l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w1"), l0), set.Get("b1")))
				l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w2"), l1), set.Get("b2")))
				l3 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w3"), l2), set.Get("b3")))
				l4 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w4"), l3), set.Get("b4")))
				l5 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w5"), l4), set.Get("b5")))
				l6 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w6"), l5), set.Get("b6")))
				l7 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w7"), l6), set.Get("b7")))
				q := m.Mix()
				copy(input.X, q[:])
				l7(func(a *tf32.V) bool {
					aa := a.X[:len(symbols)]
					sum := float32(0.0)
					for _, v := range aa {
						sum += v
					}
					selection, total := rng.Float32(), float32(0.0)
					for i, v := range aa {
						total += v / sum
						if selection < total {
							vector.Vectors[n] = Vector{
								Vector: q,
								Symbol: i,
							}
							m.Add(byte(i))
							return true
						}
					}
					return true
				})

				for l := n + 1; l < *FlagCount; l++ {
					vector.Vectors[l].Vector = m.Mix()
					m.Add(byte(vector.Vectors[l].Symbol))
				}
				vectors = append(vectors, vector)
			}
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
			return vectors[i].Rank > vectors[j].Rank
		})
		vectors = vectors[:128]
		fmt.Println(vectors[0].Rank)
	}

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

	if *FlagVDB {
		var m Mix
		if *FlagMixer == "filtered" {
			m = NewFiltered()
		} else {
			m = NewMixer()
		}

		type Vector struct {
			Vector [InputSize]float32
			Counts [256]uint32
		}
		var vectors [8 * 1024]Vector
		for i := range vectors {
			for j := range vectors[i].Vector {
				vectors[i].Vector[j] = rng.Float32()
			}
			l := sqrt(vector.Dot(vectors[i].Vector[:], vectors[i].Vector[:]))
			for j := range vectors[i].Vector {
				vectors[i].Vector[j] /= l
			}
		}

		m.Add(0)
		type Result struct {
			Index  int
			Symbol byte
		}
		done := make(chan Result, 8)
		cpus := runtime.NumCPU()
		process := func(v byte, query [InputSize]float32) {
			max, index := float32(0.0), 0
			for i := range vectors {
				cs := CS(query[:], vectors[i].Vector[:])
				if cs > max {
					max, index = cs, i
				}
			}
			done <- Result{
				Index:  index,
				Symbol: v,
			}
		}
		j, flight := 0, 0
		for j < len(in) && flight < cpus {
			query := m.Mix()
			v := in[j]
			go process(v, query)
			j++
			flight++
			m.Add(v)
			fmt.Println(float64(j) / float64(len(in)))
		}
		for j < len(in) {
			result := <-done
			for i := range vectors[result.Index].Counts {
				if vectors[result.Index].Counts[i] == math.MaxUint32 {
					for j := range vectors[result.Index].Counts {
						vectors[result.Index].Counts[j] /= 2
					}
					break
				}
			}
			vectors[result.Index].Counts[result.Symbol]++
			flight--

			query := m.Mix()
			v := in[j]
			go process(v, query)
			j++
			flight++
			m.Add(v)
			fmt.Println(float64(j)/float64(len(in)), result.Index)
		}
		for k := 0; k < flight; k++ {
			result := <-done
			for i := range vectors[result.Index].Counts {
				if vectors[result.Index].Counts[i] == math.MaxUint32 {
					for j := range vectors[result.Index].Counts {
						vectors[result.Index].Counts[j] /= 2
					}
					break
				}
			}
			vectors[result.Index].Counts[result.Symbol]++
		}
		return
	}

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

		var m Mix
		if *FlagMixer == "filtered" {
			m = NewFiltered()
		} else {
			m = NewMixer()
		}
		m.Add(0)
		buffer32 := make([]byte, 4)
		for i, v := range in[:len(in)-2] {
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
			n, err := db.Write(in[i : i+3])
			if err != nil {
				panic(err)
			}
			if n != 3 {
				panic("3 bytes should be been written")
			}
			m.Add(v)
		}
		return
	}

	type Vector struct {
		Vector [InputSize]float32
		Symbol [3]byte
	}
	vectors, err := os.Open("vectors.bin")
	if err != nil {
		panic(err)
	}
	defer vectors.Close()
	get := func(i int) Vector {
		_, err := vectors.Seek(int64((InputSize*4+3)*i), io.SeekStart)
		if err != nil {
			panic(err)
		}
		vector := Vector{}
		buffer := make([]byte, InputSize*4+3)
		n, err := vectors.Read(buffer)
		if n != len(buffer) {
			panic(fmt.Sprintf("%d bytes should have been read, %d was read", len(buffer), n))
		}
		if err != nil {
			panic(err)
		}
		for k := range vector.Vector {
			var bits uint32
			for l := 0; l < 4; l++ {
				bits |= uint32(buffer[4*k+l]) << (8 * l)
			}
			vector.Vector[k] = math.Float32frombits(bits)
		}
		vector.Symbol[0] = buffer[len(buffer)-3]
		vector.Symbol[1] = buffer[len(buffer)-2]
		vector.Symbol[2] = buffer[len(buffer)-1]
		return vector
	}

	others := tf32.NewSet()
	others.Add("input", InputSize, BatchSize)
	others.Add("output", 3*len(symbols), BatchSize)
	input, output := others.ByName["input"], others.ByName["output"]
	input.X = input.X[:cap(input.X)]
	output.X = output.X[:cap(output.X)]
	set := tf32.NewSet()
	set.Add("w0", InputSize, 8*len(symbols))
	set.Add("b0", 8*len(symbols))
	set.Add("w1", 16*len(symbols), 8*len(symbols))
	set.Add("b1", 8*len(symbols))
	set.Add("w2", 16*len(symbols), 8*len(symbols))
	set.Add("b2", 8*len(symbols))
	set.Add("w3", 16*len(symbols), 8*len(symbols))
	set.Add("b3", 8*len(symbols))
	set.Add("w4", 16*len(symbols), 8*len(symbols))
	set.Add("b4", 8*len(symbols))
	set.Add("w5", 16*len(symbols), 8*len(symbols))
	set.Add("b5", 8*len(symbols))
	set.Add("w6", 16*len(symbols), 8*len(symbols))
	set.Add("b6", 8*len(symbols))
	set.Add("w7", 16*len(symbols), 3*len(symbols))
	set.Add("b7", 3*len(symbols))
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
	l4 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w4"), l3), set.Get("b4")))
	l5 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w5"), l4), set.Get("b5")))
	l6 := tf32.Everett(tf32.Add(tf32.Mul(set.Get("w6"), l5), set.Get("b6")))
	l7 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get("w7"), l6), set.Get("b7")))
	loss := tf32.Avg(tf32.Quadratic(l7, others.Get("output")))
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
			vector := get(rng.Intn(len(in) - 2))
			copy(input.X[j*InputSize:(j+1)*InputSize], vector.Vector[:])
			for k := 0; k < 3*len(symbols); k++ {
				output.X[j*3*len(symbols)+k] = 0
			}
			output.X[j*3*len(symbols)+int(vector.Symbol[0])] = 1
			output.X[j*3*len(symbols)+len(symbols)+int(vector.Symbol[1])] = 1
			output.X[j*3*len(symbols)+2*len(symbols)+int(vector.Symbol[2])] = 1
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
