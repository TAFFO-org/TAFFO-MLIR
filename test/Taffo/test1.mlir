// RUN: taffo-opt %s | taffo-opt | FileCheck %s
// potential example for thesis
module {
    func.func @fibonacci(%n: index) -> i32 {

        %const = arith.constant dense<[[1, 1], [1, 0]]> : tensor<2x2xi32>

        %lb = index.constant 0
        // %ub = index.castu %n : i32 to index
        %ub = index.constant 5
        %st = index.constant 1

        %pow_next = arith.constant dense<0> : tensor<2x2xi32>

        %pow = scf.for %iv = %lb to %ub step %st
            iter_args(%pow_iter = %const) -> (tensor<2x2xi32>) {

            // %pow_next = linalg.matmul %pow_iter, %const : tensor<2x2xi32>

            %t = linalg.matmul
                   ins(%pow_iter, %const : tensor<2x2xi32>, tensor<2x2xi32>)
                   outs(%pow_next : tensor<2x2xi32>) -> tensor<2x2xi32>

            scf.yield %t : tensor<2x2xi32>
        }

        %zero = index.constant 0
        %res = tensor.extract %pow[%zero, %zero] : tensor<2x2xi32>

        return %res : i32
    }
}
