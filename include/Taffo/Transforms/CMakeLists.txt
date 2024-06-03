set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Taffo)
add_public_tablegen_target(MLIRTaffoTransformsIncGen)

add_mlir_doc(Passes TaffoPasses ./ -gen-pass-doc)