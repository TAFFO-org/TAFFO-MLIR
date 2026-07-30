// RUN: taffo-opt --vra-mode=interval -pass-pipeline='builtin.module(value-range-analysis)' --verify-each %s -o /dev/null
// RUN: taffo-opt --vra-mode=affine -pass-pipeline='builtin.module(value-range-analysis)' --verify-each %s -o /dev/null
// RUN: taffo-opt --vra-mode=mixed -pass-pipeline='builtin.module(value-range-analysis)' --verify-each %s | FileCheck %s
// CHECK: taffo.cast2real
// CHECK: taffo.add

module {
  func.func @simple_constant() -> f32 {
    %a = arith.constant 8.0 : f32
    %1 = taffo.cast2real %a, 0.1, 1.0, 8.0 : f32 -> !taffo.real
    %2 = taffo.add %1, %1 : (!taffo.real, !taffo.real) -> !taffo.real
    %3 = taffo.add %1, %2 : (!taffo.real, !taffo.real) -> !taffo.real
    %4 = taffo.cast2float %3 : !taffo.real -> f32
    return %4 : f32
  }
}
