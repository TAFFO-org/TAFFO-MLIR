#include "Taffo/Ops.h"
#include "Taffo/InferTaffoRangeNtvInterface.h"
#include "Taffo/TaffoRangeCommon.h"

using namespace mlir;
using namespace mlir::taffo;

using NtvRange = mlir::taffo::NtvRange;

void taffo::AssignOp::inferTaffoRanges(ArrayRef<NtvRange> argRanges,
                                       SetTaffoRangeFn setResultRange) {
  setResultRange(getResult(), inferAssign(argRanges));
}

void taffo::AddOp::inferTaffoRanges(ArrayRef<NtvRange> argRanges,
                                    SetTaffoRangeFn setResultRange) {
  setResultRange(getResult(), inferAdd(argRanges));
}