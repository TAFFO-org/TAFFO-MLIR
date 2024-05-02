#include "Taffo/TaffoDialect.h"

using namespace ::mlir::taffo;

#include "Taffo/TaffoDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Taffo dialect.
//===----------------------------------------------------------------------===//
namespace mlir::taffo {
  void TaffoDialect::initialize() {
    addOperations<
  #define GET_OP_LIST
  #include "Taffo/Taffo.cpp.inc"
        >();

    addTypes<
  #define GET_TYPEDEF_LIST
  #include "Taffo/TaffoTypes.cpp.inc"
        >();
  }
}
