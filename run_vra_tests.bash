#!/bin/bash

cd build
ninja
cd bin

set -x

for i in {1..6}
do 
    ./taffo-opt -pass-pipeline="builtin.module(value-range-analysis, dt-optimization, lower-to-arith, convert-func-to-llvm)" ./../../test/Taffo/test$i.mlir -debug-only=value-range-analysis | mlir-cpu-runner -e=simple_constant -entry-point-result=f32

done