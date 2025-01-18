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
