module {
  func.func @simple_constant() -> f32 {
        %a = arith.constant 0x00000009 : f32

        %1 = taffo.cast2real %a, 0.1, 1.0e-46, 1.0e-40 : f32 -> r
        %2 = taffo.cast2float %1 : r -> f32

        return %2 : f32
  }
}