// RUN: taffo-opt --verify-each %s -o /dev/null

func.func @sum() -> f32 {
  // Lower bound
  %lb = arith.constant 0 : index

  // Upper bound
  %ub = arith.constant 3 : index

  // Step
  %step = arith.constant 1 : index

  // Constants
  %a = arith.constant -0.75 : f32
  %x = arith.constant 0.0 : f32

  // Convert to taffo.real
  %r_a = taffo.cast2real %a, 0.1, -0.75, -0.75 : f32 -> !taffo.real
  %r_x = taffo.cast2real %x, 0.1, -1.0, 1.0 : f32 -> !taffo.real

  // Bind the initial value to the loop's region argument
  %res = scf.for %iv = %lb to %ub step %step iter_args(%iter = %r_x) -> (!taffo.real) {
    %tmp = taffo.mult %r_a, %iter : (!taffo.real, !taffo.real) -> !taffo.real
    %next = taffo.add %iter, %tmp : (!taffo.real, !taffo.real) -> !taffo.real
    scf.yield %next : !taffo.real
  }

  %final_res = taffo.cast2float %res : !taffo.real -> f32
  return %final_res : f32
}
