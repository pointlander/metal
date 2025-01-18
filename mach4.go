// Copyright 2025 The Metal Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/tar"
	"archive/zip"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// Mach4 is the mach 4 version
func Mach4() {
	if *FlagBuild {
		if *FlagInput == "" {
			panic("-input required")
		}
		archive, err := zip.OpenReader(*FlagInput)
		if err != nil {
			panic(err)
		}
		defer archive.Close()

		rng := rand.New(rand.NewSource(1))
		var model []Vector
		for _, f := range archive.File {
			fileInArchive, err := f.Open()
			if err != nil {
				panic(err)
			}
			tarReader := tar.NewReader(fileInArchive)
			for {
				header, err := tarReader.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					fmt.Println("Error reading header:", err)
					return
				}

				switch header.Typeflag {
				case tar.TypeReg:
					fmt.Println("File:", header.Name)
					buffer, err := ioutil.ReadAll(tarReader)
					if err != nil {
						panic(err)
					}
					for i := 0; i < 7; i++ {
						m, index := NewMixer(), rng.Intn(len(buffer)-(128+1))
						for _, v := range buffer[index : index+128] {
							m.Add(v)
						}
						vector64 := m.Mix().Sum().Data
						vector32 := make([]float32, len(vector64))
						for i, v := range vector64 {
							vector32[i] = float32(v)
						}
						model = append(model, Vector{
							Vector: vector32,
							Symbol: buffer[index+128],
						})
					}
				case tar.TypeDir:
					fmt.Println("Directory:", header.Name)
				default:
					fmt.Println("Unknown type:", header.Typeflag)
				}
			}
			fileInArchive.Close()
		}
		db, err := os.Create("vdb.bin")
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
			n, err := db.Write([]byte{model[i].Symbol})
			if err != nil {
				panic(err)
			}
			if n != 1 {
				panic("1 byte should be been written")
			}
		}

		return
	}

	var model []Vector
	in, err := os.Open("vdb.bin")
	if err != nil {
		panic(err)
	}
	defer in.Close()
	buffer, symbol := make([]byte, 4), make([]byte, 1)
	for {
		vector := Vector{
			Vector: make([]float32, 256),
		}
		for j := range vector.Vector {
			_, err := in.Read(buffer)
			if err != nil {
				break
			}
			bits := uint32(buffer[0])
			bits |= uint32(buffer[1]) << 8
			bits |= uint32(buffer[2]) << 16
			bits |= uint32(buffer[3]) << 24
			vector.Vector[j] = math.Float32frombits(bits)
		}
		_, err := in.Read(symbol)
		if err != nil {
			break
		}
		vector.Symbol = symbol[0]
		model = append(model, vector)
	}

	m := NewMixer()
	for _, v := range *FlagQuery {
		m.Add(byte(v))
	}

	for i := 0; i < 77; i++ {
		vector := m.Mix().Sum().Data
		index, max := 0, 0.0
		for j := range model {
			cs := CSFloat64(model[j].Vector, vector)
			if cs > max {
				max, index = cs, j
			}
		}
		y := strconv.Quote(string(model[index].Symbol))
		y = strings.TrimRight(strings.TrimLeft(y, "\""), "\"")
		fmt.Printf(y)
		m.Add(byte(model[index].Symbol))
	}
}
