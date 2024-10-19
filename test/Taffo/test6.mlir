module {
  func.func @simple_constant() -> f32 {
    %cst1 = arith.constant 1.5 : f32
    %cst2 = arith.constant 2.5 : f32
    %cst3 = arith.constant -1.0 : f32
    %0 = taffo.cast2real %cst1, 1.000000e-01, -1.0, 2.5 : f32 -> r
    %1 = taffo.cast2real %cst2, 1.000000e-01, -1.0, 2.5 : f32 -> r
    %neg1 = taffo.cast2real %cst3, 1.000000e-01, -1.0, 2.5 : f32 -> r
    %neg0 = taffo.mult %0, %neg1: (r, r) -> r
    %res = taffo.add %0, %neg0 : (r, r) -> r
    %2 = taffo.mult %0, %1 : (r, r) -> r
    %3 = taffo.cast2float %2 : r -> f32
    %4 = taffo.add %0, %1 : (r, r) -> r
    return %3 : f32
  }
}