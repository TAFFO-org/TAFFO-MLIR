#include "Taffo/Dialect/Ops.h"
#include "Taffo/Dialect/Taffo.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
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

namespace mlir::taffo {

void CastToRealOp::inferTaffoRanges(
    llvm::ArrayRef<NtvRange> argRanges,
    mlir::taffo::SetTaffoRangeFn setResultRange) {
  // Since CastToRealOp's ranges are inferred from its attributes (NOT its
  // operands!), we can discard ArgRanges and use accessors methods instead
  setResultRange(getResult(), NtvRange(getMin(), getMax()));
}

void CastToFloatOp::inferTaffoRanges(
    llvm::ArrayRef<NtvRange> argRanges,
    mlir::taffo::SetTaffoRangeFn setResultRange) {
  setResultRange(getResult(), inferCastToFloat(argRanges));
}

void AddOp::inferTaffoRanges(llvm::ArrayRef<NtvRange> argRanges,
                             mlir::taffo::SetTaffoRangeFn setResultRange) {
  setResultRange(getResult(), inferAdd(argRanges));
}

void MultOp::inferTaffoRanges(llvm::ArrayRef<NtvRange> argRanges,
                              mlir::taffo::SetTaffoRangeFn setResultRange) {
  setResultRange(getResult(), inferMult(argRanges));
}

void WideAddOp::inferTaffoRanges(llvm::ArrayRef<NtvRange> argRanges,
                              mlir::taffo::SetTaffoRangeFn setResultRange) {
  setResultRange(getResult(), inferAdd(argRanges));
}

void WideMultOp::inferTaffoRanges(llvm::ArrayRef<NtvRange> argRanges,
                              mlir::taffo::SetTaffoRangeFn setResultRange) {
  setResultRange(getResult(), inferMult(argRanges));
}

void BitcastOp::inferTaffoRanges(llvm::ArrayRef<NtvRange> argRanges,
                                 mlir::taffo::SetTaffoRangeFn setResultRange) {
  setResultRange(getResult(), NtvRange(llvm::APFloat(0.0), llvm::APFloat(0.0)));
}

} // namespace mlir::taffo
