// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        %1 = arith.constant 1 : i4
        // CHECK: %{{.*}} = standalone.foo %{{.*}} : i32
        %2 = arith.addi %0, %1
        %res = standalone.foo %0 : i32
        return
    }
}
