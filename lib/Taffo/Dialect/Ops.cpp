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

#include "Taffo/Interfaces/InferTaffoRangeInterface.h"
#include "Taffo/Transforms/AffineRangeAnalysis.hpp"
#include "Taffo/Transforms/TaffoRangeCommon.h"
#include "llvm/ADT/APFloat.h"
using namespace ::mlir;
using namespace ::mlir::taffo;
using namespace LibAffine;

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

void SubOp::inferTaffoRanges(llvm::ArrayRef<NtvRange> argRanges,
                             mlir::taffo::SetTaffoRangeFn setResultRange) {
  setResultRange(getResult(), inferSub(argRanges));
}

void CastToRealOp::inferTaffoAffineRanges(
    llvm::ArrayRef<Var> argRanges,
    mlir::taffo::SetTaffoAffineRangeFn setResultRange) {
  // Since CastToRealOp's ranges are inferred from its attributes (NOT its
  // operands!), we can discard ArgRanges and use accessors methods instead
  setResultRange(getResult(), Var(LibAffine::Range(getMin(), getMax())));
}

void CastToFloatOp::inferTaffoAffineRanges(
    llvm::ArrayRef<Var> argRanges,
    mlir::taffo::SetTaffoAffineRangeFn setResultRange) {
  setResultRange(getResult(), inferCastToFloat(argRanges));
}

void AddOp::inferTaffoAffineRanges(
    llvm::ArrayRef<Var> argRanges,
    mlir::taffo::SetTaffoAffineRangeFn setResultRange) {
  setResultRange(getResult(), inferAdd(argRanges));
}

void MultOp::inferTaffoAffineRanges(
    llvm::ArrayRef<Var> argRanges,
    mlir::taffo::SetTaffoAffineRangeFn setResultRange) {
  setResultRange(getResult(), inferMult(argRanges));
}

void SubOp::inferTaffoAffineRanges(
    llvm::ArrayRef<Var> argRanges,
    mlir::taffo::SetTaffoAffineRangeFn setResultRange) {
  setResultRange(getResult(), inferSub(argRanges));
}

} // namespace mlir::taffo
