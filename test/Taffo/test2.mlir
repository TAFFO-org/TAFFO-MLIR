// RUN: taffo-opt %s | taffo-opt | FileCheck %s

module {
    func.func @simple_constant()  {

        %1 = arith.constant 10.3 : f32
        %2 = arith.constant 4.5 : f32
        %3 = arith.constant 0.01 : f32

        %4 = taffo.assign %1, %2, %3
        %5 = taffo.assign %1, %2, %3
        %6 = taffo.add %4, %5

        return
    }
}
