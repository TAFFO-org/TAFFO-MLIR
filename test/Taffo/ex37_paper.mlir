func.func @sum() -> (f32) {

  // lower bound
  %lb = arith.constant 0 : index

  // upper bound
  %ub = arith.constant 3 : index

  //step
  %step = arith.constant 1 : index


  //Define constant
  %a = arith.constant -0.75: f32
  
  %x = arith.constant 0.0: f32

  //Convert to taffo.real
  %r_a = taffo.cast2real %a, 0.1, -0.75, -0.75 : f32 -> !taffo.real
  %r_x = taffo.cast2real %x, 0.1, -1.0, 1.0 : f32 -> !taffo.real


  // iter_args binds initial values to the loop's region arguments.
  %res = scf.for %iv = %lb to %ub step %step
      iter_args(%iter = %r_x) -> (!taffo.real) {
      
    %tmp = taffo.mult %r_a, %iter : (!taffo.real, !taffo.real) -> !taffo.real
    %next = taffo.add %iter, %tmp : (!taffo.real, !taffo.real) -> !taffo.real

    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    scf.yield %next : !taffo.real
  }

  %final_res = taffo.cast2float %res : !taffo.real -> f32

  return %final_res : f32
}