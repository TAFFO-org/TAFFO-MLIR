#include "Taffo/Dialect/TaffoDialect.h"
#include "Taffo/Dialect/Types.h"

using namespace ::mlir::taffo;

#include "Taffo/Dialect/TaffoDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Taffo dialect.
//===----------------------------------------------------------------------===//
namespace mlir::taffo {
  void TaffoDialect::initialize() {
    registerTypes();

    addOperations<
  #define GET_OP_LIST
  #include "Taffo/Dialect/Taffo.cpp.inc"
        >();
  }
}
