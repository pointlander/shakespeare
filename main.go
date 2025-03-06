// Copyright 2025 The Shakespeare Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"fmt"
	"io"
)

//go:embed books/*
var Data embed.FS

func main() {
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
	symbols, s := make(map[rune]int), 0
	for _, v := range string(data) {
		if _, has := symbols[v]; !has {
			symbols[v] = s
			s++
		}
	}
	fmt.Println(s)
}
