// RUN: taffo-opt --verify-each %s -o /dev/null

func.func @reduce() -> f32 {
  // Lower bound
  %lb = arith.constant 0 : index

  // Upper bound
  %ub = arith.constant 12 : index

  // Step
  %step = arith.constant 1 : index

  // Initial sum
  %sum_0 = arith.constant 1.2 : f32
  %r_sum_0 = taffo.cast2real %sum_0, 0.1, 0.0, 2.0 : f32 -> !taffo.real

  // Initial product
  %prod_0 = arith.constant 1.2 : f32
  %r_prod_0 = taffo.cast2real %sum_0, 0.1, 0.0, 2.0 : f32 -> !taffo.real

  // Bind the initial values to the loop's region arguments
  %sum, %prod = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %r_sum_0, %prod_iter = %r_prod_0) -> (!taffo.real, !taffo.real) {
    %sum_next = taffo.add %sum_iter, %r_sum_0 : (!taffo.real, !taffo.real) -> !taffo.real
    %prod_next = taffo.add %prod_iter, %r_prod_0 : (!taffo.real, !taffo.real) -> !taffo.real
    scf.yield %sum_next, %prod_next : !taffo.real, !taffo.real
  }

  %comb = taffo.add %sum, %prod : (!taffo.real, !taffo.real) -> !taffo.real
  %res = taffo.cast2float %comb : !taffo.real -> f32
  return %res : f32
}
