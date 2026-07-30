// RUN: taffo-opt --vra-mode=mixed -pass-pipeline='builtin.module(value-range-analysis)' --verify-each %s -o /dev/null

module {
  func.func @simple_constant() -> f32 {
    %a = arith.constant 1001.0 : f32
    %1 = taffo.cast2real %a, 0.1, 1001.0, 1001.0 : f32 -> !taffo.real
    %2 = taffo.cast2float %1 : !taffo.real -> f32
    return %2 : f32
  }
}
