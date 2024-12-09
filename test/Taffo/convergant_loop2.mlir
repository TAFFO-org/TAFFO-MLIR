module {
  func.func @reduce() -> (f32) {
  %lb = arith.constant 0 : index

  %ub = arith.constant 10 : index

  %step = arith.constant 1 : index

  %mult = arith.constant 0.2: f32
  // CHECK: %0 = taffo.cast2real {{.*}}, 1.000000e-01, 2.000000e-01, 2.000000e-01 : f32 -> <exponent = -26, bitwidth = 24>
  %r_mult = taffo.cast2real %mult, 0.1, 0.2, 0.2 : f32 -> !taffo.real

  %sum_0 = arith.constant 1.5 : f32
  // CHECK: %1 = taffo.cast2real {{.*}}, 1.000000e-01, 1.000000e+00, 2.000000e+01 : f32 -> <exponent = -19, bitwidth = 24>
  %r_sum_0 = taffo.cast2real %sum_0, 0.1, 1.0, 20.0 : f32 -> !taffo.real

  %sum_1 = arith.constant 1.5 : f32
  // CHECK: taffo.cast2real {{.*}}, 1.000000e-01, 1.000000e+00, 2.000000e+01 : f32 -> <exponent = -19, bitwidth = 24>
  %r_sum_1 = taffo.cast2real %sum_1, 0.1, 1.0, 20.0 : f32 -> !taffo.real

  // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!taffo.real<exponent = -19, bitwidth = 24>, !taffo.real<exponent = -19, bitwidth = 24>) {
  %sum, %sum1 = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %r_sum_0, %sum_iter1 = %r_sum_1) -> (!taffo.real, !taffo.real) {
    // CHECK: %5 = taffo.mult {{.*}} : (<exponent = -19, bitwidth = 24>, <exponent = -26, bitwidth = 24>) -> <exponent = -19, bitwidth = 24>
    %tmp = taffo.mult %sum_iter, %r_mult : (!taffo.real, !taffo.real) -> !taffo.real
    // CHECK: %6 = taffo.mult {{.*}} : (<exponent = -19, bitwidth = 24>, <exponent = -26, bitwidth = 24>) -> <exponent = -19, bitwidth = 24>
    %tmp1 = taffo.mult %sum_iter1, %r_mult : (!taffo.real, !taffo.real) -> !taffo.real
    // CHECK: scf.yield {{.*}} : !taffo.real<exponent = -19, bitwidth = 24>, !taffo.real<exponent = -19, bitwidth = 24>
    scf.yield %tmp, %tmp1 : !taffo.real, !taffo.real
  }

  // CHECK: taffo.cast2float {{.*}} : <exponent = -19, bitwidth = 24> -> f32
  %res = taffo.cast2float %sum : !taffo.real -> f32
  return %res : f32
  }
}