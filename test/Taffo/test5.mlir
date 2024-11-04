module {
  func.func @simple_constant() -> f32 {
    %cst1 = arith.constant 1.5 : f32
    %cst2 = arith.constant 2.5 : f32
    %0 = taffo.cast2real %cst1, 1.000000e-01, -1.0, 2.5 : f32 -> !taffo.real
    %1 = taffo.cast2real %cst2, 1.000000e-01, -1.0, 2.5 : f32 -> !taffo.real
    %2 = taffo.mult %0, %1 : (!taffo.real, !taffo.real) -> !taffo.real
    %3 = taffo.cast2float %2 : !taffo.real -> f32
    return %3 : f32
  }
}