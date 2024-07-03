#include "Taffo/Dialect/Ops.h"
#include "Taffo/Dialect/Taffo.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"

#include "Taffo/Interfaces/InferTaffoRangeNtvInterface.h"
#include "Taffo/Transforms/TaffoRangeCommon.h"
#include "llvm/ADT/APFloat.h"

using namespace ::mlir;
using namespace ::mlir::taffo;

#define GET_OP_CLASSES
#include "Taffo/Dialect/TaffoOps.cpp.inc"

namespace mlir::taffo
{

  void CastOp::inferTaffoRanges(
      llvm::ArrayRef<std::pair<llvm::APFloat, llvm::APFloat>> argRanges,
      mlir::taffo::SetTaffoRangeFn setResultRange)
  {
    setResultRange(getResult(), inferCast(argRanges));
  }

  void AddOp::inferTaffoRanges(
    llvm::ArrayRef<std::pair<llvm::APFloat, llvm::APFloat>> argRanges,
    mlir::taffo::SetTaffoRangeFn setResultRange)
  {
    setResultRange(getResult(), inferAdd(argRanges));
  }
}
