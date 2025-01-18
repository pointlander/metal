// Copyright 2025 The Metal Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math/rand"
	"sort"
	"strconv"
)

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
