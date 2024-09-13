module {
  func.func @simple_constant() -> f32 {
    %cst1 = arith.constant 1.5 : f32
    %cst2 = arith.constant -0.9 : f32
    %0 = taffo.cast2real %cst1, 1.000000e-01, 0.0, 18000.0 : f32 -> r
    %1 = taffo.cast2real %cst2, 1.000000e-01, 0.0, 18000.0 : f32 -> r
    %2 = taffo.wmult %0, %1 : (r, r) -> r
    %3 = taffo.cast2float %2 : r -> f32
    return %3 : f32
  }
}