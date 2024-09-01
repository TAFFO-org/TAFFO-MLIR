func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>) -> (memref<128x128xf32>) attributes {llvm.emit_c_interface} {
  %f0 = arith.constant 0.0 : f32
  %c = arith.constant 0.0 : f32
  %C = memref.alloc() : memref<128x128xf32>
  linalg.fill ins(%f0 : f32) outs(%C : memref<128x128xf32>)
  linalg.matmul ins(%A, %B: memref<128x128xf32>, memref<128x128xf32>)
                outs(%C: memref<128x128xf32>)
  return %C : memref<128x128xf32>
}
