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

#include "Taffo/Dialect/OpInterfaces.h"
#include "Taffo/Transforms/TaffoRangeCommon.h"
#include "llvm/ADT/APFloat.h"

using namespace ::mlir;
using namespace ::mlir::taffo;

#define GET_OP_CLASSES
#include "Taffo/Dialect/TaffoOps.cpp.inc"

namespace mlir::taffo {

LogicalResult CastToRealOp::verify() {
  if (getMin() > getMax()) {
    emitOpError("Lower bound must be less than or equal to upper bound");
    return failure();
  }

  if (getMin() == getMax() &&
      !getFrom().getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>()) {
    emitOpError(
        "Lower bound and upper bound coincide but $from is not a constant");
    return failure();
  }

  // need support for other ConstantLike-s?
  auto const_op =
      ::llvm::dyn_cast<arith::ConstantOp>(getFrom().getDefiningOp());

  auto from = ::llvm::dyn_cast<FloatAttr>(const_op.getValue()).getValue();

  if (getMin() == getMax() &&
      from.convertToDouble() != getMax().convertToDouble()) {
    emitOpError(
        "Lower bound and upper bound coincide but are not equal to $from");
    return failure();
  }

  return success();
}

LogicalResult AlignOp::verify() {
  RealType source = getFrom().getType();
  RealType target = getRes().getType();
  auto getMSB = [](RealType t) { return t.getBitwidth() + t.getExponent(); };

  if (getMSB(source) > getMSB(target)) {
    emitOpError(
        "Target MSB weight must be greater or equal to source MSB weight");
    return failure();
  }
  if (target.getSignd() != source.getSignd()) {
    emitOpError("Target sign must be equal to source sign");
    return failure();
  }
  return success();
}

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

} // namespace mlir::taffo
