// Copyright 2025 The Metal Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math/rand"
	"strconv"
)

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
