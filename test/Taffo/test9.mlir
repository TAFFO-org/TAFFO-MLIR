module {
  func.func @simple_constant() -> f32 {
    %cst0 = arith.constant 1.5 : f32
    %cst1 = arith.constant 1.5 : f32
    %cst2 = arith.constant 2.0 : f32

    %flag = arith.constant 1 : i1  // Placeholder for random value

    %0 = taffo.cast2real %cst0, 1.000000e-01, 1.0, 3.0 : f32 -> !taffo.real
    %1 = taffo.cast2real %cst1, 1.000000e-01, 1.0, 3.0 : f32 -> !taffo.real
    %2 = taffo.cast2real %cst2, 1.000000e-01, 2.0, 2.0 : f32 -> !taffo.real

    // Branch based on a random condition
    %3, %4 = scf.if %flag -> (!taffo.real, !taffo.real) {
      %rt1 = taffo.add %0, %2 : (!taffo.real, !taffo.real) -> !taffo.real
      %rt2 = taffo.add %1, %2 : (!taffo.real, !taffo.real) -> !taffo.real
      scf.yield %rt1, %rt2 : !taffo.real, !taffo.real

    } else {
      scf.yield %0, %1 : !taffo.real, !taffo.real
    }
    
    %tmp = taffo.sub %3, %4 : (!taffo.real, !taffo.real) -> !taffo.real
    %final = taffo.cast2float %tmp : !taffo.real -> f32
    return %final : f32
  }
}
