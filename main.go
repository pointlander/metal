// Copyright 2025 The Metal Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"embed"
	"flag"
	"fmt"
	"math"
)

const (
	// Size is the number of histograms
	Size = 8
	// Order is the order of the markov model
	Order = 7
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

// Matrix is a float64 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float64
}

// NewMatrix creates a new float32 matrix
func NewMatrix(cols, rows int, data ...float64) Matrix {
	if data == nil {
		data = make([]float64, 0, cols*rows)
	}
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix) MulT(n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// Add adds two float32 matrices
func (m Matrix) Add(n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Softmax calculates the softmax of the matrix rows
func (m Matrix) Softmax(T float64) Matrix {
	output := NewMatrix(m.Cols, m.Rows)
	max := 0.0
	for _, v := range m.Data {
		v /= T
		if v > max {
			max = v
		}
	}
	s := max * S
	for i := 0; i < len(m.Data); i += m.Cols {
		sum := 0.0
		values := make([]float64, m.Cols)
		for j, value := range m.Data[i : i+m.Cols] {
			values[j] = math.Exp(value/T - s)
			sum += values[j]
		}
		for _, value := range values {
			output.Data = append(output.Data, value/sum)
		}
	}
	return output
}

// Entropy calculates the entropy of the matrix rows
func (m Matrix) Entropy() Matrix {
	output := NewMatrix(m.Rows, 1)
	for i := 0; i < len(m.Data); i += m.Cols {
		entropy := 0.0
		for _, value := range m.Data[i : i+m.Cols] {
			entropy += value * math.Log(value)
		}
		output.Data = append(output.Data, -entropy)
	}
	return output
}

// Sum sums the rows of a matrix
func (m Matrix) Sum() Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: 1,
		Data: make([]float64, m.Cols),
	}
	for i := 0; i < m.Rows; i++ {
		offset := i * m.Cols
		for j := range o.Data {
			o.Data[j] += m.Data[offset+j]
		}
	}
	return o
}

// T tramsposes a matrix
func (m Matrix) T() Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

func dot(x, y []float64) (z float64) {
	for i := range x {
		z += x[i] * y[i]
	}
	return z
}

func softmax(values []float64) {
	max := 0.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	for j, value := range values {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(Q, K, V Matrix) Matrix {
	o := Matrix{
		Cols: V.Cols,
		Rows: K.Rows,
		Data: make([]float64, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]float64, V.Cols), make([]float64, Q.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = dot(values, V)
		}
		o.Data = append(o.Data, outputs...)
	}
	return o
}

// Markov is a markov model
type Markov [Order + 1]byte

// Histogram is a buffered histogram
type Histogram struct {
	Vector [256]byte
	Buffer [128]byte
	Index  int
	Size   int
}

// NewHistogram make a new histogram
func NewHistogram(size int) Histogram {
	h := Histogram{
		Size: size,
	}
	return h
}

// Add adds a symbol to the histogram
func (h *Histogram) Add(s byte) {
	index := (h.Index + 1) % h.Size
	if symbol := h.Buffer[index]; h.Vector[symbol] > 0 {
		h.Vector[symbol]--
	}
	h.Buffer[index] = s
	h.Vector[s]++
	h.Index = index
}

// Mixer mixes several histograms together
type Mixer struct {
	Markov     Markov
	Histograms []Histogram
}

// NewMixer makes a new mixer
func NewMixer() Mixer {
	histograms := make([]Histogram, Size)
	histograms[0] = NewHistogram(1)
	histograms[1] = NewHistogram(2)
	histograms[2] = NewHistogram(4)
	histograms[3] = NewHistogram(8)
	histograms[4] = NewHistogram(16)
	histograms[5] = NewHistogram(32)
	histograms[6] = NewHistogram(64)
	histograms[7] = NewHistogram(128)
	return Mixer{
		Histograms: histograms,
	}
}

func (m Mixer) Copy() Mixer {
	histograms := make([]Histogram, Size)
	for i := range m.Histograms {
		histograms[i] = m.Histograms[i]
	}
	return Mixer{
		Markov:     m.Markov,
		Histograms: histograms,
	}
}

// Add adds a symbol to a mixer
func (m *Mixer) Add(s byte) {
	for i := range m.Histograms {
		m.Histograms[i].Add(s)
	}
	for k := Order; k > 0; k-- {
		m.Markov[k] = m.Markov[k-1]
	}
	m.Markov[0] = s
}

// Mix mixes the histograms outputting a matrix
func (m Mixer) Mix() Matrix {
	x := NewMatrix(256, Size)
	for i := range m.Histograms {
		sum := 0.0
		for _, v := range m.Histograms[i].Vector {
			sum += float64(v)
		}
		for _, v := range m.Histograms[i].Vector {
			x.Data = append(x.Data, float64(v)/sum)
		}
	}
	y := SelfAttention(x, x, x)
	return y
}

//go:embed books/*
var Data embed.FS

var (
	// FlagMach1 is the mach 1 version
	FlagMach1 = flag.Bool("mach1", false, "mach 1 version")
	// FlagMach2 is the mach 2 version
	FlagMach2 = flag.Bool("mach2", false, "mach 2 version")
	// FlagMach3 is the mach 3 version
	FlagMach3 = flag.Bool("mach3", false, "mach 3 version")
	// FlagMach4 is the mach 4 version
	FlagMach4 = flag.Bool("mach4", false, "mach 4 version")
	// FlagMach5 is the mach 5 version
	FlagMach5 = flag.Bool("mach5", false, "mach 5 version")
	// FlagQuery is the query string
	FlagQuery = flag.String("query", "What is the meaning of life?", "query flag")
	// FlagBuild build the database
	FlagBuild = flag.Bool("build", false, "build the database")
	// FlagInput is the input for building
	FlagInput = flag.String("input", "", "input for building")
)

// Vector is a vector
type Vector struct {
	Vector []float32
	Symbol byte
}

// Bucket is a bucket of vectors
type Bucket struct {
	Vector  [256]float32
	Vectors []Vector
}

// Pair is a pair of values
type Pair struct {
	Symbol byte
	Rank   float64
}

// CS is float32 cosine similarity
func CS(t []float32, vector []float64) float32 {
	aa, bb, ab := 0.0, 0.0, 0.0
	for i := range vector {
		a, b := float64(vector[i]), float64(t[i])
		aa += a * a
		bb += b * b
		ab += a * b
	}
	return float32(ab / (math.Sqrt(aa) * math.Sqrt(bb)))
}

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
	// Scale is the scale of the model
	Scale = 128
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Entry is a vector database entry
type Entry struct {
	Vector [256]float32
	Counts [256]uint32
}

// CSFloat64 is float64 cosine similarity
func CSFloat64(t []float32, vector []float64) float64 {
	aa, bb, ab := 0.0, 0.0, 0.0
	for i := range vector {
		a, b := vector[i], float64(t[i])
		aa += a * a
		bb += b * b
		ab += a * b
	}
	return ab / (math.Sqrt(aa) * math.Sqrt(bb))
}

func main() {
	flag.Parse()

	if *FlagMach1 {
		Mach1()
		return
	} else if *FlagMach2 {
		Mach2()
		return
	} else if *FlagMach3 {
		Mach3()
		return
	} else if *FlagMach4 {
		Mach4()
		return
	} else if *FlagMach5 {
		Mach5()
		return
	}
}
