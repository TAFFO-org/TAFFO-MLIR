get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRTaffo
)

add_llvm_executable(taffo-opt taffo-opt.cpp)

set(ONNX_MLIR_SRC_DIR "" CACHE PATH "Path to ONNX-MLIR source directory")
set(ONNX_MLIR_BUILD_DIR "" CACHE PATH "Path to ONNX-MLIR build directory")

if(ONNX_MLIR_SRC_DIR AND ONNX_MLIR_BUILD_DIR)
  target_include_directories(taffo-opt PRIVATE
    ${ONNX_MLIR_SRC_DIR}
    ${ONNX_MLIR_BUILD_DIR}
  )
endif()

llvm_update_compile_flags(taffo-opt)

target_link_libraries(taffo-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(taffo-opt)
