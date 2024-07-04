// RUN: taffo-opt %s | taffo-opt | FileCheck %s

module {
    func.func @simple_constant()  {

        %a = arith.constant 0.5 : f64
        %b = arith.constant 6.7 : f64

        %1 = arith.constant 0.01 : f64
        %2 = arith.constant 4.5  : f64
        %3 = arith.constant 10.3 : f64

        //%5 = taffo.cast %b, %1, %2, %3     : !taffo.real
        //%4 = taffo.cast %a, 0.1, -1.0, 1.0 : f32 -> !taffo.real
        %4 = taffo.cast 0.1, -1.0, 1.0 : !taffo.real

        %6 = taffo.add %4, %4 : !taffo.real

        return
    }
}
