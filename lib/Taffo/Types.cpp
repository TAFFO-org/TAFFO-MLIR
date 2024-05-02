#include "Taffo/Types.h"
#include "Taffo/TaffoDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/APFloat.h"


using namespace ::mlir::taffo;

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Taffo/TaffoTypes.cpp.inc"