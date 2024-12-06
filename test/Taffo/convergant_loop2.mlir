func.func @reduce() -> (f32) {

  // lower bound
  %lb = arith.constant 0 : index

  // upper bound
  %ub = arith.constant 10 : index

  //step
  %step = arith.constant 1 : index

  //Constant multiplier
  %mult = arith.constant 0.2: f32
  %r_mult = taffo.cast2real %mult, 0.1, 0.2, 0.2 : f32 -> !taffo.real
  // Initial sum set to 0.
  %sum_0 = arith.constant 1.5 : f32
  %r_sum_0 = taffo.cast2real %sum_0, 0.1, 1.0, 20.0 : f32 -> !taffo.real

  %sum_1 = arith.constant 1.5 : f32
  %r_sum_1 = taffo.cast2real %sum_1, 0.1, 1.0, 20.0 : f32 -> !taffo.real

  // iter_args binds initial values to the loop's region arguments.
  %sum, %sum1 = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %r_sum_0, %sum_iter1 = %r_sum_1) -> (!taffo.real, !taffo.real) {

    %tmp = taffo.mult %sum_iter, %r_mult : (!taffo.real, !taffo.real) -> !taffo.real
    %tmp1 = taffo.mult %sum_iter1, %r_mult : (!taffo.real, !taffo.real) -> !taffo.real

    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    scf.yield %tmp, %tmp1 : !taffo.real, !taffo.real
  }

  %res = taffo.cast2float %sum : !taffo.real -> f32

  return %res : f32
}