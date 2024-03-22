add_mlir_dialect(TaffoOps taffo)
add_mlir_doc(TaffoOps TaffoDialect Taffo/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TaffoOps.td)
mlir_tablegen(TaffoOps.h.inc -gen-op-decls)
mlir_tablegen(TaffoOps.cpp.inc -gen-op-defs)
#add_public_tablegen_target(MLIRTaffoOpsIncGen)
#add_dependencies(MLIRTaffoIncGen MLIRTaffoOpsIncGen)