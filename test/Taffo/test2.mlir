// RUN: taffo-opt %s | taffo-opt | FileCheck %s

module {
    func.func @simple_constant() -> (f32)  {

        %a = arith.constant 2.7 : f32

        %1 = taffo.cast2real %a, 0.1, 1.0, 10000.0 : f32 -> r
        %2 = taffo.add %1, %1 : (r, r) -> r
        %3 = taffo.add %1, %2 : (r, r) -> r
        %4 = taffo.cast2float %3 : r -> f32

        return %4 : f32
    }
}
