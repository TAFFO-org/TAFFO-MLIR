#include "Taffo/TaffoDialect.h"
#include "Taffo/TaffoOps.h"

using namespace mlir;
using namespace mlir::taffo;

//===----------------------------------------------------------------------===//
// Taffo dialect.
//===----------------------------------------------------------------------===//

void TaffoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Taffo/TaffoOps.cpp.inc"
      >();
}
