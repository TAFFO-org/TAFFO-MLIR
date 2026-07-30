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

  // Bind the initial value to the loop's region argument
  %sum = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %r_sum_0) -> (!taffo.real) {
    %sum_next = taffo.add %sum_iter, %r_sum_0 : (!taffo.real, !taffo.real) -> !taffo.real
    scf.yield %sum_next : !taffo.real
  }

  %res = taffo.cast2float %sum : !taffo.real -> f32
  return %res : f32
}
