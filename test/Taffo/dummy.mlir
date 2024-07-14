// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %1 = arith.constant 160 : i8               // %1 is 0b10100000
        %3 = arith.shrui %1, %2 : (i8, i8) -> i8   // %3 is 0b11110100
        %2 = arith.constant 3 : i8
        %4 = arith.constant 96 : i8                   // %4 is 0b01100000
        %5 = arith.shrsi %4, %2 : (i8, i8) -> i8
        return
    }
}