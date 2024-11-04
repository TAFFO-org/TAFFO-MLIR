module {
  func.func @simple_constant() -> f32 {
    %cst1 = arith.constant 1.5 : f32
    %cst2 = arith.constant 2.5 : f32
   
    %0 = taffo.cast2real %cst1, 1.000000e-01, -1.0, 2.5 : f32 -> r
    %1 = taffo.cast2real %cst2, 1.000000e-01, 2.5, 2.5 : f32 -> r

    %res = taffo.add %0, %1 : (r, r) -> r
    %res1 = taffo.add %0, %res : (r, r) -> r
    %res2 = taffo.add %0, %res1 : (r, r) -> r
  
    %res3 = taffo.sub %res2, %0 : (r, r) -> r

    %2 = taffo.mult %0, %res3 : (r, r) -> r

    %3 = taffo.cast2float %2 : r -> f32
    return %3 : f32
  }
}