func.func @main() {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %c1_2 = arith.constant 1.2 : f32
    %c0_8 = arith.constant -0.8 : f32

    // Initial values
    %xn = taffo.cast2real %c0, 0.1, 0.0, 1.0 : f32 -> !taffo.real
    %r_c1_2 = taffo.cast2real %c1_2, 0.1, 1.2, 1.2 : f32 -> !taffo.real
    %r_c0_8 = taffo.cast2real %c0_8, 0.1, -0.8, -0.8 : f32 -> !taffo.real

    // Iteration 1
    %xnp1_1 = taffo.mult %r_c1_2, %xn : (!taffo.real, !taffo.real) -> !taffo.real
    // %xnm1_1 = %xn : (!taffo.real) -> !taffo.real
    // %xn_1 = %xnp1_1 : !taffo.real

    // Iteration 2
    %tmp1 = taffo.mult %r_c0_8, %xn : (!taffo.real, !taffo.real) -> !taffo.real
    %tmp2 = taffo.mult %r_c1_2, %xnp1_1 : (!taffo.real, !taffo.real) -> !taffo.real
    %xnp1_2 = taffo.add %tmp1, %tmp2 : (!taffo.real, !taffo.real) -> !taffo.real
    // %xnm1_2 = %xn_1 : !taffo.real
    // %xn_2 = %xnp1_2 : !taffo.real

    // Iteration 3
    %tmp3 = taffo.mult %r_c0_8, %xnp1_1 : (!taffo.real, !taffo.real) -> !taffo.real
    %tmp4 = taffo.mult %r_c1_2, %xnp1_2 : (!taffo.real, !taffo.real) -> !taffo.real
    %xnp1_3 = taffo.add %tmp3, %tmp4 : (!taffo.real, !taffo.real) -> !taffo.real
    // %xnm1_3 = %xn_2 : !taffo.real
    // %xn_3 = %xnp1_3 : !taffo.real

    // Final values (optional use for validation/output)
    // %final_xn = %xnp1_3 : !taffo.real

    return
}
