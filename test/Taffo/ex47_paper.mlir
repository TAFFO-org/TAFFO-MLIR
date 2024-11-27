
  func.func @ex47_paper() -> f32 {
    %cst0 = arith.constant 0.0 : f32
    %cst1 = arith.constant 1.0 : f32
    %cst_half = arith.constant 0.5 : f32
    %cst_neg1 = arith.constant -1.0 : f32

    %z = taffo.cast2real %cst1, 1.000000e-01, 0.0, 1.0 : f32 -> !taffo.real

    %0 = taffo.cast2real %cst0, 1.000000e-01, 0.0, 1.0 : f32 -> !taffo.real
    %1 = taffo.cast2real %cst1, 1.000000e-01, 0.0, 1.0 : f32 -> !taffo.real
    %neg1 = taffo.cast2real %cst_neg1, 1.000000e-01, -1.0, 1.0 : f32 -> !taffo.real
    %half = taffo.cast2real %cst_half, 1.000000e-01, 0.0, 1.0 : f32 -> !taffo.real

    %cmp1 = arith.cmpf ult, %cst1, %cst_half : f32
    %cmp2 = arith.cmpf ugt, %cst1, %cst0 : f32

    %x, %y = scf.if %cmp1 -> (!taffo.real, !taffo.real) {
      scf.yield %1, %neg1 : !taffo.real, !taffo.real
    } else {
      scf.yield %0, %1 : !taffo.real, !taffo.real
    }

    %x_final = scf.if %cmp2 -> !taffo.real {
      %x_new = taffo.add %x, %y : (!taffo.real, !taffo.real) -> !taffo.real
      scf.yield %x_new : !taffo.real
    } else {
      scf.yield %x : !taffo.real
    }

    %res = taffo.cast2float %x_final : !taffo.real -> f32
    return %res : f32
  }
