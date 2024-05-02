#include "Taffo/TaffoDialect.h"
#include "Taffo/Types.h"

using namespace ::mlir::taffo;

#include "Taffo/TaffoDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Taffo dialect.
//===----------------------------------------------------------------------===//
namespace mlir::taffo {
  void TaffoDialect::initialize() {
    registerTypes();

    addOperations<
  #define GET_OP_LIST
  #include "Taffo/Taffo.cpp.inc"
        >();
  }
}
