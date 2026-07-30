// RUN: taffo-opt --verify-each %s -o /dev/null

func.func @sum() -> f32 {
  // Lower bound
  %lb = arith.constant 1 : index

  // Upper bound
  %ub = arith.constant 3 : index

  // Step
  %step = arith.constant 1 : index

  // Constant two
  %const_two = arith.constant 2.0 : f32
  %r_const_two = taffo.cast2real %const_two, 0.1, 2.0, 2.0 : f32 -> !taffo.real

  // Initial sum
  %sum_0 = arith.constant 0.0 : f32
  %r_sum_0 = taffo.cast2real %sum_0, 0.1, 0.0, 2.0 : f32 -> !taffo.real

  // Bind the initial value to the loop's region argument
  %sum = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %r_sum_0) -> (!taffo.real) {
    %sum_next = taffo.add %sum_iter, %r_const_two : (!taffo.real, !taffo.real) -> !taffo.real
    scf.yield %sum_next : !taffo.real
  }

  %res = taffo.cast2float %sum : !taffo.real -> f32
  return %res : f32
}
