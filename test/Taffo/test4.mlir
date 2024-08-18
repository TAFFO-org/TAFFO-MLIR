// RUN: taffo-opt %s | taffo-opt | FileCheck %s

module {
    func.func @simple_constant() -> (f32)  {

        %a = arith.constant 0.6 : f32

        %1 = taffo.cast2real %a, 0.1, -1.0, 10.0 : f32 -> !taffo.real
        %2 = taffo.cast2float %1 : !taffo.real -> f32

        return %2 : f32
    }
}
