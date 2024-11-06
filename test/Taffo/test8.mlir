module {
  func.func @simple_constant() -> f32 {
    %cst1 = arith.constant 1.5 : f32
    %cst2 = arith.constant 2.5 : f32
    %random = arith.constant 0 : i1  // Placeholder for random value

    %0 = taffo.cast2real %cst1, 1.000000e-01, -1.0, 2.5 : f32 -> !taffo.real
    %1 = taffo.cast2real %cst2, 1.000000e-01, 0.0, 1.0 : f32 -> !taffo.real

    %res = taffo.add %0, %1 : (!taffo.real, !taffo.real) -> !taffo.real
    %res1 = taffo.add %0, %res : (!taffo.real, !taffo.real) -> !taffo.real
    %res2 = taffo.add %0, %res1 : (!taffo.real, !taffo.real) -> !taffo.real

    // Branch based on a random condition
    cf.cond_br %random, ^true_branch, ^false_branch

  ^true_branch:
    %res3 = taffo.sub %res2, %0 : (!taffo.real, !taffo.real) -> !taffo.real
    %result_true = taffo.mult %0, %res3 : (!taffo.real, !taffo.real) -> !taffo.real
    cf.br ^continue(%result_true : !taffo.real)

  ^false_branch:
    %result_false = taffo.mult %0, %res2 : (!taffo.real, !taffo.real) -> !taffo.real
    cf.br ^continue(%result_false : !taffo.real)

  ^continue(%final_result: !taffo.real):
    %3 = taffo.cast2float %final_result : !taffo.real -> f32
    return %3 : f32
  }
}
