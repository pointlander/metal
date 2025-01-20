// Copyright 2025 The Metal Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
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

// Mach5 is the mach 5 mode
func Mach5() {
	const ModelSize = 8
	cpus := runtime.NumCPU()
	file, err := Data.Open("books/10.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	input, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}
	data := input

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
		model := [ModelSize * 1024]Bucket{}
		fmt.Println(ModelSize * 1024 * 512 * 4.0 / (1024.0 * 1024.0 * 1024.0))
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
			Index  int
			Vector []float64
			Symbol byte
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
				Index:  index,
				Vector: query,
				Symbol: symbol,
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
			vec := make([]float32, len(result.Vector))
			for i, v := range result.Vector {
				vec[i] = float32(v)
			}
			model[result.Index].Vectors = append(model[result.Index].Vectors, Vector{
				Vector: vec,
				Symbol: result.Symbol,
			})

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
			vec := make([]float32, len(result.Vector))
			for i, v := range result.Vector {
				vec[i] = float32(v)
			}
			model[result.Index].Vectors = append(model[result.Index].Vectors, Vector{
				Vector: vec,
				Symbol: result.Symbol,
			})
		}

		db, err := os.Create("tdb.bin")
		if err != nil {
			panic(err)
		}
		defer db.Close()

		buffer32 := make([]byte, 4)
		buffer64 := make([]byte, 8)
		for i := range model {
			for _, v := range model[i].Vector {
				bits := math.Float32bits(v)
				for i := 0; i < 4; i++ {
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
			count := uint64(len(model[i].Vectors))
			for i := 0; i < 8; i++ {
				buffer64[i] = byte((count >> (8 * i)) & 0xFF)
			}
			n, err := db.Write(buffer64)
			if err != nil {
				panic(err)
			}
			if n != len(buffer64) {
				panic("8 bytes should be been written")
			}
		}

		symbol := make([]byte, 1)
		for i := range model {
			for _, vector := range model[i].Vectors {
				for _, v := range vector.Vector {
					bits := math.Float32bits(v)
					for i := 0; i < 4; i++ {
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
				symbol[0] = vector.Symbol
				n, err := db.Write(symbol)
				if err != nil {
					panic(err)
				}
				if n != len(symbol) {
					panic("1 bytes should be been written")
				}
			}
		}
		return
	}

	model := [ModelSize * 1024]Bucket{}
	sizes := make([]uint64, ModelSize*1024)
	in := make([]*os.File, 4)
	for i := range in {
		var err error
		in[i], err = os.Open("tdb.bin")
		if err != nil {
			panic(err)
		}
	}
	defer func() {
		for i := range in {
			in[i].Close()
		}
	}()
	buffer32 := make([]byte, 4)
	buffer64 := make([]byte, 8)
	for i := range model {
		for j := range model[i].Vector {
			n, err := in[0].Read(buffer32)
			if err != nil {
				panic(err)
			}
			if n != len(buffer32) {
				panic("4 bytes should have been read")
			}
			var bits uint32
			for i := range buffer32 {
				bits |= uint32(buffer32[i]) << (8 * i)
			}
			model[i].Vector[j] = math.Float32frombits(bits)
		}
		var count uint64
		n, err := in[0].Read(buffer64)
		if err != nil {
			panic(err)
		}
		if n != len(buffer64) {
			panic("4 bytes should have been read")
		}
		for i := range buffer64 {
			count |= uint64(buffer64[i]) << (8 * i)
		}
		sizes[i] = count
	}
	sums, sum := make([]uint64, len(sizes)), uint64(0)
	for i, v := range sizes {
		sums[i] = sum
		sum += v
	}

	m := NewMixer()
	for _, v := range []byte(*FlagQuery) {
		m.Add(v)
	}

	type Index struct {
		Index int
		Value float32
	}
	type Result struct {
		Symbol byte
		Max    float32
	}
	const offset = ModelSize * 1024 * (4*256 + 1*8)
	done := make(chan Result, 8)
	search := func(r, index int, data []float64) {
		max, symbol := float32(0.0), byte(0)
		buffer32, vector, buffer8 := make([]byte, 4), make([]float32, 256), make([]byte, 1)
		_, err := in[r].Seek(int64(offset+sums[index]*(4*256+1)), io.SeekStart)
		if err != nil {
			panic(err)
		}
		for j := 0; j < int(sizes[index]); j++ {
			for j := range vector {
				n, err := in[r].Read(buffer32)
				if err != nil {
					panic(err)
				}
				if n != len(buffer32) {
					panic("4 bytes should have been read")
				}
				var bits uint32
				for k := range buffer32 {
					bits |= uint32(buffer32[k]) << (8 * k)
				}
				vector[j] = math.Float32frombits(bits)
			}
			n, err := in[r].Read(buffer8)
			if err != nil {
				panic(err)
			}
			if n != len(buffer8) {
				panic("1 byte should have been read")
			}
			cs := CS(vector, data)
			if cs > max {
				max, symbol = cs, buffer8[0]
			}
		}
		done <- Result{
			Symbol: symbol,
			Max:    max,
		}
	}
	sample := func(m Mixer) string {
		result := make([]byte, 0, 8)
		for i := 0; i < 128; i++ {
			data := m.Mix().Sum().Data
			indexes := make([]Index, len(model))
			for i := range model {
				if sizes[i] == 0 {
					continue
				}
				indexes[i].Index = i
				indexes[i].Value = CS(model[i].Vector[:], data)
			}
			sort.Slice(indexes, func(i, j int) bool {
				return indexes[i].Value > indexes[j].Value
			})

			var results []Result
			for j := 0; j < 4; j++ {
				go search(j, indexes[j].Index, data)
			}
			for j := 0; j < 4; j++ {
				result := <-done
				results = append(results, result)
			}
			sort.Slice(results, func(i, j int) bool {
				return results[i].Max > results[j].Max
			})

			symbol := results[0].Symbol
			fmt.Printf("%d %s\n", symbol, strconv.Quote(string(symbol)))
			m.Add(symbol)
			result = append(result, symbol)
		}
		return string(result)
	}
	result := sample(m.Copy())
	fmt.Println(result)
}
