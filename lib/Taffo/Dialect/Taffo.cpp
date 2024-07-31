#include "Taffo/Dialect/Taffo.h"
#include "Taffo/Dialect/Types.h"
#include "Taffo/Dialect/Attributes.h"

using namespace ::mlir::taffo;

#include "Taffo/Dialect/Taffo.cpp.inc"

//===----------------------------------------------------------------------===//
// Taffo dialect.
//===----------------------------------------------------------------------===//
namespace mlir::taffo {
  void TaffoDialect::initialize() {
    registerTypes();
    registerAttributes();

    addOperations<
  #define GET_OP_LIST
  #include "Taffo/Dialect/TaffoOps.cpp.inc"
        >();
  }
}
