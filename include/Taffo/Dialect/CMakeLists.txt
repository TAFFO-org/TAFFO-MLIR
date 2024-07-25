#add_mlir_dialect(Taffo taffo)
#add_mlir_doc(TaffoOps TaffoDialect Taffo/ -gen-dialect-doc)

add_mlir_doc(Taffo Taffo Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TaffoAttributes.td)
mlir_tablegen(TaffoAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TaffoAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTaffoAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS TaffoTypes.td)
mlir_tablegen(TaffoTypes.h.inc -gen-typedef-decls -typedefs-dialect=taffo)
mlir_tablegen(TaffoTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=taffo)
add_public_tablegen_target(MLIRTaffoTypesIncGen)

set(LLVM_TARGET_DEFINITIONS TaffoOps.td)
mlir_tablegen(TaffoOps.h.inc -gen-op-decls)
mlir_tablegen(TaffoOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTaffoOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Taffo.td)
mlir_tablegen(Taffo.h.inc -gen-dialect-decls -dialect=taffo)
mlir_tablegen(Taffo.cpp.inc -gen-dialect-defs -dialect=taffo)
add_public_tablegen_target(MLIRTaffoIncGen)

add_dependencies(MLIRTaffoIncGen MLIRTaffoAttributesIncGen)
add_dependencies(MLIRTaffoIncGen MLIRTaffoTypesIncGen)
add_dependencies(MLIRTaffoIncGen MLIRTaffoOpsIncGen)