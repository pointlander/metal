// Copyright 2025 The Metal Authors. All rights reserved.
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
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf32"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
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
	// FlagQuery is the query string
	FlagQuery = flag.String("query", "What is the meaning of life?", "query flag")
	// FlagBuild build the database
	FlagBuild = flag.Bool("build", false, "build the database")
)

// Pair is a pair of values
type Pair struct {
	Symbol byte
	Rank   float64
}

// Mach1 is the mach 1 model
func Mach1() {
	rng := rand.New(rand.NewSource(1))
	file, err := Data.Open("84.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	data, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}
	valid := make(map[byte]bool)
	for _, v := range data {
		valid[v] = true
	}
	fmt.Println("valid", len(valid))

	vdb := make(map[Markov][256]int)
	m := NewMixer()
	for _, v := range data {
		vector := m.Mix()
		distro := vector.Sum().Softmax(1)
		pairs := make([]Pair, distro.Cols)
		for i, v := range distro.Data {
			pairs[i].Symbol = byte(i)
			pairs[i].Rank = v
		}
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Rank > pairs[j].Rank
		})
		markov := Markov{}
		for i := range markov {
			markov[i] = pairs[i].Symbol
		}
		dist := vdb[markov]
		dist[v]++
		vdb[markov] = dist
		for j := Order; j > 0; j-- {
			markov[j] = 0
			dist = vdb[markov]
			dist[v]++
			vdb[markov] = dist
		}
		m.Add(v)
	}
	fmt.Println("len(vdb)", len(vdb))

	m = NewMixer()
	for _, v := range []byte(*FlagQuery) {
		m.Add(v)
	}

	result := make([]byte, 0, 8)
	for j := 0; j < 33; j++ {
		output := m.Mix()
		distro := output.Sum().Softmax(1)
		pairs := make([]Pair, distro.Cols)
		for i, v := range distro.Data {
			pairs[i].Symbol = byte(i)
			pairs[i].Rank = v
		}
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Rank > pairs[j].Rank
		})
		markov := Markov{}
		for i := range markov {
			markov[i] = pairs[i].Symbol
		}
		dist, found := vdb[markov]
		for i := Order; i > 0 && !found; i-- {
			markov[i] = 0
			dist, found = vdb[markov]
		}
		values, sum := make([]float64, len(dist)), 0.0
		for _, v := range dist {
			sum += float64(v)
		}
		for i, v := range dist {
			values[i] = float64(v) / sum
		}
		sum, symbol, selection := 0.0, 0, rng.Float64()
		for i, v := range values {
			sum += v
			if selection < sum {
				symbol = i
				break
			}
		}
		fmt.Printf("%d %s\n", symbol, strconv.Quote(string(byte(symbol))))
		m.Add(byte(symbol))
		result = append(result, byte(symbol))
	}
	fmt.Println(string(result))
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

// Mach2 is the mach 2 mode
func Mach2() {
	cpus := runtime.NumCPU()
	books := []string{
		"books/10.txt.utf-8.bz2",
		"books/84.txt.utf-8.bz2",
		"books/2701.txt.utf-8.bz2",
	}
	var data []byte
	for _, book := range books {
		file, err := Data.Open(book)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		reader := bzip2.NewReader(file)
		input, err := io.ReadAll(reader)
		if err != nil {
			panic(err)
		}
		data = append(data, input...)
	}

	if *FlagBuild {
		rng := rand.New(rand.NewSource(1))

		avg := make([]float64, 256)
		m := NewMixer()
		m.Add(0)
		for _, v := range data {
			vector := m.Mix().Sum()
			for i, v := range vector.Data {
				avg[i] += v
			}
			m.Add(v)
		}
		for i := range avg {
			avg[i] /= float64(len(data))
		}
		cov := [256][256]float64{}
		m = NewMixer()
		m.Add(0)
		for _, v := range data {
			vector := m.Mix().Sum()
			for i, v := range vector.Data {
				for ii, vv := range vector.Data {
					diff1 := avg[i] - v
					diff2 := avg[ii] - vv
					cov[i][ii] += diff1 * diff2
				}
			}
			m.Add(v)
		}
		for i := range cov {
			for j := range cov[i] {
				cov[i][j] = cov[i][j] / float64(len(data))
			}
		}
		fmt.Println(avg)

		set := tf32.NewSet()
		set.Add("A", 256, 256)

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

		others := tf32.NewSet()
		others.Add("E", 256, 256)
		E := others.ByName["E"]
		for i := range cov {
			for j := range cov[i] {
				E.X = append(E.X, float32(cov[i][j]))
			}
		}

		loss := tf32.Sum(tf32.Quadratic(others.Get("E"), tf32.Mul(set.Get("A"), set.Get("A"))))

		points := make(plotter.XYs, 0, 8)
		for i := 0; i < 1024; i++ {
			pow := func(x float32) float32 {
				y := math.Pow(float64(x), float64(i+1))
				if math.IsNaN(y) || math.IsInf(y, 0) {
					return 0
				}
				return float32(y)
			}

			set.Zero()
			others.Zero()
			cost := tf32.Gradient(loss).X[0]
			if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
				fmt.Println(i, cost)
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
			points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
			fmt.Println(i, cost)
		}

		p := plot.New()

		p.Title.Text = "epochs vs cost"
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
		if err != nil {
			panic(err)
		}

		A := NewMatrix(256, 256)
		for _, v := range set.ByName["A"].X {
			A.Data = append(A.Data, float64(v))
		}
		u := NewMatrix(256, 1, avg...)
		model := [Scale * 1024]Entry{}
		fmt.Println(Scale * 1024 * 512 * 4.0 / (1024.0 * 1024.0 * 1024.0))
		for i := range model {
			z := NewMatrix(256, 1)
			for j := 0; j < 256; j++ {
				z.Data = append(z.Data, rng.NormFloat64())
			}
			x := A.MulT(z).Add(u)
			for j, v := range x.Data {
				model[i].Vector[j] = float32(v)
			}
		}

		type Result struct {
			Symbol byte
			Index  int
		}
		done := make(chan Result, 8)
		process := func(symbol byte, vector Matrix) {
			query := vector.Sum().Data
			index, max := 0, float32(0.0)
			for i := range model {
				cs := CS(model[i].Vector[:], query)
				if cs > max {
					max, index = cs, i
				}
			}
			done <- Result{
				Symbol: symbol,
				Index:  index,
			}
		}

		m, index, flight := NewMixer(), 0, 0
		m.Add(0)
		for index < len(data) && flight < cpus {
			symbol := data[index]
			vector := m.Mix()
			go process(symbol, vector)
			m.Add(symbol)
			flight++
			index++
		}
		for index < len(data) {
			result := <-done
			flight--
			model[result.Index].Counts[result.Symbol]++

			symbol := data[index]
			vector := m.Mix()
			go process(symbol, vector)
			m.Add(symbol)
			flight++
			index++
			if index%8 == 0 {
				fmt.Println(index, "/", len(data), "=", float64(index)/float64(len(data)))
			}
		}
		for i := 0; i < flight; i++ {
			result := <-done
			model[result.Index].Counts[result.Symbol]++
		}

		db, err := os.Create("db.bin")
		if err != nil {
			panic(err)
		}
		defer db.Close()
		buffer := make([]byte, 4)
		for i := range model {
			for _, v := range model[i].Vector {
				bits := math.Float32bits(v)
				buffer[0] = byte(bits & 0xFF)
				buffer[1] = byte((bits >> 8) & 0xFF)
				buffer[2] = byte((bits >> 16) & 0xFF)
				buffer[3] = byte((bits >> 24) & 0xFF)
				n, err := db.Write(buffer)
				if err != nil {
					panic(err)
				}
				if n != len(buffer) {
					panic("4 bytes should be been written")
				}
			}
			for _, v := range model[i].Counts {
				bits := v
				buffer[0] = byte(bits & 0xFF)
				buffer[1] = byte((bits >> 8) & 0xFF)
				buffer[2] = byte((bits >> 16) & 0xFF)
				buffer[3] = byte((bits >> 24) & 0xFF)
				n, err := db.Write(buffer)
				if err != nil {
					panic(err)
				}
				if n != len(buffer) {
					panic("4 bytes should be been written")
				}
			}

		}
		return
	}

	rng := rand.New(rand.NewSource(1))

	model := [Scale * 1024]Entry{}
	in, err := os.Open("db.bin")
	if err != nil {
		panic(err)
	}
	defer in.Close()
	buffer := make([]byte, 4)
	for i := range model {
		for j := range model[i].Vector {
			n, err := in.Read(buffer)
			if err != nil {
				panic(err)
			}
			if n != len(buffer) {
				panic("4 bytes should have been read")
			}
			bits := uint32(buffer[0])
			bits |= uint32(buffer[1]) << 8
			bits |= uint32(buffer[2]) << 16
			bits |= uint32(buffer[3]) << 24
			model[i].Vector[j] = math.Float32frombits(bits)
		}
		for j := range model[i].Counts {
			n, err := in.Read(buffer)
			if err != nil {
				panic(err)
			}
			if n != len(buffer) {
				panic("4 bytes should have been read")
			}
			bits := uint32(buffer[0])
			bits |= uint32(buffer[1]) << 8
			bits |= uint32(buffer[2]) << 16
			bits |= uint32(buffer[3]) << 24
			model[i].Counts[j] = bits
		}

	}

	m := NewMixer()
	for _, v := range []byte(*FlagQuery) {
		m.Add(v)
	}

	sample := func(m Mixer) (int, string) {
		value, result := 0, make([]byte, 0, 8)
		for i := 0; i < 33; i++ {
			vector := m.Mix()
			distro := vector.Sum()
			index, max := 0, float32(0.0)
			for i := range model {
				isZero := true
				for _, v := range model[i].Counts {
					if v != 0 {
						isZero = false
						break
					}
				}
				if isZero {
					continue
				}
				cs := CS(model[i].Vector[:], distro.Data)
				if cs > max {
					max, index = cs, i
				}
			}
			x := NewMatrix(len(model[index].Counts), 1)
			for _, v := range model[index].Counts {
				value += int(v)
				x.Data = append(x.Data, float64(v))
			}
			x = x.Softmax(1)
			sum, selected, symbol := float32(0.0), rng.Float32(), 0
			for i, v := range x.Data {
				sum += float32(v)
				if selected < sum {
					symbol = i
					break
				}
			}

			fmt.Printf("%d %s\n", symbol, strconv.Quote(string(byte(symbol))))
			m.Add(byte(symbol))
			result = append(result, byte(symbol))
		}
		return value, string(result)
	}
	type Result struct {
		Result string
		Value  int
	}
	results := make([]Result, 10)
	for i := range results {
		value, result := sample(m.Copy())
		results[i].Value = value
		results[i].Result = result
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Value > results[j].Value
	})
	for _, r := range results {
		fmt.Println(r.Value, r.Result)
		fmt.Println()
	}
}

// Mach3 is the mach 3 mode
func Mach3() {
	file, err := Data.Open("84.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	data, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}

	model := [64 * 1024][512]float32{}
	fmt.Println(64 * 1024 * 512 * 4.0 / (1024.0 * 1024.0 * 1024.0))
	m := NewMixer()
	m.Add(0)
	for i, v := range data[:len(data)-1] {
		vector := m.Mix().Sum()
		context := ((uint(v) << 8) | (uint(data[i+1]) & 0xFF)) & 0xFFFF
		for i := range model[context][:256] {
			model[context][i] += float32(vector.Data[i])
		}
		model[context][256+uint(v)]++
		m.Add(v)
	}

	rng := rand.New(rand.NewSource(1))

	m = NewMixer()
	for _, v := range []byte(*FlagQuery) {
		m.Add(v)
	}

	result := make([]byte, 0, 8)
	for i := 0; i < 33; i++ {
		vector := m.Mix()
		distro := vector.Sum()
		index, max := 0, float32(0.0)
		for i := range model {
			cs := CS(model[i][:256], distro.Data)
			if cs > max {
				max, index = cs, i
			}
		}
		x := NewMatrix(len(model[index]), 1)
		for _, v := range model[index] {
			x.Data = append(x.Data, float64(v))
		}
		x = x.Softmax(1)
		sum, selected, symbol := float32(0.0), rng.Float32(), 0
		for i, v := range x.Data {
			sum += float32(v)
			if selected < sum {
				symbol = i
				break
			}
		}

		fmt.Printf("%d %s\n", symbol, strconv.Quote(string(byte(symbol))))
		m.Add(byte(symbol))
		result = append(result, byte(symbol))
	}
	fmt.Println(string(result))
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
	}
}
