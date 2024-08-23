// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
func.func @fibonacci (%n : i32) -> (i32) {
  // First two fibonacci numbers

  %f = arith.constant dense<[0, 1]> : vector<2xi32>

  // lower bound
  %lb = arith.constant 1 : index

  // upper bound
  %ub = index.castu %n : i32 to index

  //step
  %step = arith.constant 1 : index

  // iter_args binds initial values to the loop's region arguments.
  %fib = scf.for %iv = %lb to %ub step %step
      iter_args(%fib_iter = %f) -> (vector<2xi32>) {

    %first  = vector.extract %fib_iter[0] : i32 from vector<2xi32>
    %second = vector.extract %fib_iter[1] : i32 from vector<2xi32>

    %fib_next = arith.addi %first, %second : i32

    %empty = arith.constant dense<[0, 0]> : vector<2xi32>
    %next_iter_b  = vector.insert %second, %empty[0] : i32 into vector<2xi32>
    %next_iter = vector.insert %fib_next, %empty[1] : i32 into vector<2xi32>

    scf.yield %next_iter : vector<2xi32>
  }
  %res = vector.extract %fib[1] : i32 from vector<2xi32>
  return %res : i32
}
}