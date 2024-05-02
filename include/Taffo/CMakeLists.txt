add_mlir_dialect(Taffo taffo)
add_mlir_doc(TaffoOps TaffoDialect Taffo/ -gen-dialect-doc)

#set(LLVM_TARGET_DEFINITIONS TaffoOps.td)
#mlir_tablegen(TaffoOps.h.inc -gen-op-decls)
#mlir_tablegen(TaffoOps.cpp.inc -gen-op-defs)

#set(LLVM_TARGET_DEFINITIONS TaffoTypes.td)
#mlir_tablegen(TaffoTypes.h.inc -gen-op-decls)
#mlir_tablegen(TaffoTypes.cpp.inc -gen-op-defs)

#add_public_tablegen_target(MLIRTaffoOpsIncGen)
#add_dependencies(MLIRTaffoIncGen MLIRTaffoOpsIncGen)