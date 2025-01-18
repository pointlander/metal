// Copyright 2025 The Metal Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/tar"
	"archive/zip"
	"compress/bzip2"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Mach4 is the mach 4 version
func Mach4() {
	archive, err := zip.OpenReader("/home/andrew/share/txt-files.tar.zip")
	if err != nil {
		panic(err)
	}
	defer archive.Close()

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
				// buf, _ := ioutil.ReadAll(tarReader)
			case tar.TypeDir:
				fmt.Println("Directory:", header.Name)
			default:
				fmt.Println("Unknown type:", header.Typeflag)
			}
		}
		fileInArchive.Close()
	}

	books := []string{
		"books/10.txt.utf-8.bz2",
		"books/84.txt.utf-8.bz2",
		"books/2701.txt.utf-8.bz2",
	}
	data := make(map[string][]byte)
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
		data[book] = input
	}
	type Vector struct {
		Vector []float32
		Symbol byte
		Next   byte
	}
	vectors := make([]Vector, 10*1024)
	sub := vectors[:len(vectors)-1]
	m := NewMixer()
	for i := range sub {
		s := data["books/10.txt.utf-8.bz2"][i]
		m.Add(byte(s))
		vector := m.Mix().Sum()
		v := make([]float32, len(vector.Data))
		for j := range v {
			v[j] = float32(vector.Data[j])
		}
		sub[i].Vector = v
		sub[i].Symbol = s
		sub[i].Next = data["books/10.txt.utf-8.bz2"][i+1]
	}

	m = NewMixer()
	query := []byte("What is the meaning of life?")
	for _, v := range query {
		m.Add(byte(v))
	}

	for i := 0; i < 33; i++ {
		vector := m.Mix().Sum().Data
		index, max := 0, 0.0
		for j := range sub {
			cs := CSFloat64(sub[j].Vector, vector)
			if cs > max {
				max, index = cs, j
			}
		}
		y := strconv.Quote(string(sub[index].Next))
		y = strings.TrimRight(strings.TrimLeft(y, "\""), "\"")
		fmt.Printf(y)
		m.Add(byte(sub[index].Next))
	}

	m = NewMixer()
	for _, v := range data["books/84.txt.utf-8.bz2"] {
		m.Add(byte(v))
		vector := m.Mix().Sum().Data
		index, max := 0, 0.0
		for j := range sub {
			cs := CSFloat64(sub[j].Vector, vector)
			if cs > max {
				max, index = cs, j
			}
		}
		x := strconv.Quote(string(v))
		x = strings.TrimRight(strings.TrimLeft(x, "\""), "\"")
		y := strconv.Quote(string(sub[index].Symbol))
		y = strings.TrimRight(strings.TrimLeft(y, "\""), "\"")
		if x != y {
			fmt.Println(x, y)
		}
	}
}
