#!/bin/bash

cd build
ninja
cd bin

set -x

for i in {1..7}
do 
    ./taffo-opt -pass-pipeline="builtin.module(value-range-analysis)" ./../../test/Taffo/test$i.mlir -debug-only=value-range-analysis

done