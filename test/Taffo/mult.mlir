// RUN: taffo-opt %s | taffo-opt | FileCheck %s

module {
    func.func @mult_test()  {

        %a = arith.constant 0.5 : f64
        %b = arith.constant 2.4 : f64

        %1 = taffo.cast %a, 0.1, -1.0, 1.0 : f64 -> !taffo.real
        %2 = taffo.cast %b, 0.1, -4.0, 3.0 : f64 -> !taffo.real
        %3 = taffo.mult %1, %2 : !taffo.real
        %4 = taffo.convert %3 : !taffo.real -> f16

        return
    }
}
