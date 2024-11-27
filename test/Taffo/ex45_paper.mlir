func.func @main() {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %c1_2 = arith.constant 1.2 : f32
    %c0_8 = arith.constant -0.8 : f32

    // lower bound
    %lb = arith.constant 0 : index

    // upper bound
    %ub = arith.constant 5 : index

    //step
    %step = arith.constant 1 : index
    
    %xn = taffo.cast2real %c0, 0.1, 0.0, 1.0 : f32 -> !taffo.real
    %xnm1 = taffo.cast2real %c0, 0.1, 0.0, 1.0 : f32 -> !taffo.real
    %xnp1 = taffo.cast2real %c0, 0.1, 0.0, 1.0 : f32 -> !taffo.real

    %r_c1_2 = taffo.cast2real %c1_2, 0.1, 1.2, 1.2 : f32 -> !taffo.real
    %r_c0_8 = taffo.cast2real %c0_8, 0.1, -0.8, -0.8 : f32 -> !taffo.real

    // Peeling the first iteration
    %xnp1_next_p = taffo.mult %r_c1_2, %xn : (!taffo.real, !taffo.real) -> !taffo.real
    // %tmp2_p = taffo.mult %r_c0_8, %xnm1 : (!taffo.real, !taffo.real) -> !taffo.real
    // %xnp1_next_p = taffo.add %tmp1, %tmp2 : (!taffo.real, !taffo.real) -> !taffo.real

    // Adjust the loop to start from the second iteration
    %lb_new = arith.constant 1 : index
    %final_xn, %final_xnm1 = scf.for %iv = %lb_new to %ub step %step
                    iter_args(%ac_xn = %xnp1_next_p, %ac_xnm1 = %xn) -> (!taffo.real, !taffo.real){
        %tmp1 = taffo.mult %r_c1_2, %ac_xn : (!taffo.real, !taffo.real) -> !taffo.real
        %tmp2 = taffo.mult %r_c0_8, %ac_xnm1 : (!taffo.real, !taffo.real) -> !taffo.real
        %xnp1_next = taffo.add %tmp1, %tmp2 : (!taffo.real, !taffo.real) -> !taffo.real
                                        
        scf.yield %xnp1_next, %ac_xn : !taffo.real, !taffo.real

    }
    
    return
}
