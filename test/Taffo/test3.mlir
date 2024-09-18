module {
  func.func @simple_constant() {
    %cst = arith.constant 5.000000e-01 : f64
    %0 = taffo.cast2real %cst, 1.000000e-01, -1.000000e+00, 1.000000e+00 : f64 -> r
    %1 = taffo.add %0, %0 : r
    %2 = taffo.add %0, %1 : r
    %3 = taffo.cast2float %2 : r -> f16
    return
  }
}