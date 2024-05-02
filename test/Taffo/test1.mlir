// RUN: taffo-opt %s | taffo-opt | FileCheck %s

module {
    func.func @simple_constant()  {

        %1 = arith.constant 4.25 : f16
        %2 = arith.fptoui %1 : f16 to i16
        %3 = arith.bitcast %1 : f16 to i16

        %4 = arith.constant 15 : i32
        %5 = arith.constant 14 : i9
        %6 = arith.extui %5 : i9 to i32
        %7 = arith.addi %4, %6 : i32

        return
    }
}
