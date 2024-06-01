#include "Taffo/Ops.h"
#include "Taffo/TaffoDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"

#include "llvm/ADT/APFloat.h"

using namespace ::mlir;
using namespace ::mlir::taffo;

#define GET_OP_CLASSES
#include "Taffo/Taffo.cpp.inc"

namespace mlir::taffo
{
  void AddOp::inferTaffoRanges(
    llvm::ArrayRef<std::pair<llvm::APFloat, llvm::APFloat>> argRanges,
    mlir::taffo::SetTaffoRangeFn setResultRanges)
  {
   llvm_unreachable("Not implemented");
  }

  void AssignOp::inferTaffoRanges(
     llvm::ArrayRef<std::pair<llvm::APFloat, llvm::APFloat>> argRanges,
     mlir::taffo::SetTaffoRangeFn setResultRanges)
  {
   llvm_unreachable("Not implemented");
  }
}
