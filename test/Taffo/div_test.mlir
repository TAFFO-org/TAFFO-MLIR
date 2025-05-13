// RUN: taffo-opt %s | taffo-opt | FileCheck %s

module {
    func.func @div_test() -> f32 {

        %a = arith.constant 0.5 : f32
        %b = arith.constant 2.4 : f32

        %1 = taffo.cast2real %a, 0.1, 0.5, 0.5 : f32 -> !taffo.real
        %2 = taffo.cast2real %b, 0.1, 0.1, 3.0 : f32 -> !taffo.real
        %3 = taffo.div %2, %1 : (!taffo.real, !taffo.real) -> !taffo.real
        %4 = taffo.cast2float %3 : !taffo.real -> f32

        return %4 : f32
    }
}
