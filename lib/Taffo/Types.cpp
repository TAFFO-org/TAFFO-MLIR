#include "Taffo/Types.h"
#include "Taffo/TaffoDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/APFloat.h"

using namespace ::mlir::taffo;
using namespace ::mlir::taffo::detail;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Taffo/TaffoTypes.cpp.inc"

//===---------------------------------------------------------------------===//
// Taffo dialect
//===---------------------------------------------------------------------===//

namespace mlir::taffo
{
  void TaffoDialect::registerTypes()
  {
    addTypes<
#define GET_TYPEDEF_LIST
#include "Taffo/TaffoTypes.cpp.inc"
        >();
  }
}
