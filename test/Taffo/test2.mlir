// RUN: taffo-opt %s | taffo-opt | FileCheck %s

module {
    func.func @simple_constant()  {

        %a = arith.constant 0.5 : f64

        %1 = taffo.cast 0.1, -1.0, 1.0 : !taffo.real
        %2 = taffo.add %1, %1 : !taffo.real
        %3 = taffo.add %1, %2 : !taffo.real

        return
    }
}
