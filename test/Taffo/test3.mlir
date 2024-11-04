module {
  func.func @simple_constant() {
    %cst = arith.constant 5.000000e-01 : f64
    %0 = taffo.cast2real %cst, 1.000000e-01, -1.000000e+00, 1.000000e+00 : f64 -> !taffo.real
    %1 = taffo.add %0, %0 : (!taffo.real, !taffo.real) -> !taffo.real
    %2 = taffo.add %0, %1 : (!taffo.real, !taffo.real) -> !taffo.real
    %3 = taffo.cast2float %2 : !taffo.real -> f16
    return
  }
}