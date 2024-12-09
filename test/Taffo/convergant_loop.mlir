// RUN: ./taffo-opt -pass-pipeline="builtin.module(value-range-analysis)" ../../test/Taffo/convergant_loop.mlir -debug-only=value-range-analysis --verify-each=1 | FileCheck ../../test/Taffo/convergant_loop.mlir

func.func @convergant_loop() -> (f32) {
  // lower bound
  %lb = arith.constant 0 : index

  // upper bound
  %ub = arith.constant 10 : index

  // step
  %step = arith.constant 1 : index

  // Constant multiplier
  %mult = arith.constant 0.1 : f32
  %r_mult = taffo.cast2real %mult, 0.1, 0.1, 0.1 : f32 -> !taffo.real

  // Initial sum set to 0.
  %sum_0 = arith.constant 1.2 : f32
  %r_sum_0 = taffo.cast2real %sum_0, 0.1, 1.0, 20.0 : f32 -> !taffo.real

  // iter_args binds initial values to the loop's region arguments.
  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %r_sum_0) -> (!taffo.real) {
    // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!taffo.real<exponent = -19, bitwidth = 24>)

    %tmp = taffo.mult %sum_iter, %r_mult : (!taffo.real, !taffo.real) -> !taffo.real
    // CHECK: taffo.mult {{.*}} : (<exponent = -19, bitwidth = 24>, <exponent = -27, bitwidth = 24>) -> <exponent = -19, bitwidth = 24>
    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    scf.yield %tmp : !taffo.real
    // CHECK: scf.yield {{.*}} : !taffo.real<exponent = -19, bitwidth = 24>
  }

  %res = taffo.cast2float %sum : !taffo.real -> f32
  return %res : f32
}