#include "Taffo/Transforms/TaffoRangeCommon.h"

using namespace ::llvm;

namespace mlir::taffo {


  template<>
  NtvRange inferCast(ArrayRef<NtvRange> argRanges) {
    assert(argRanges[0].first <= argRanges[0].second);
    return argRanges[0];
  }


  template<>
  NtvRange inferAdd(ArrayRef<NtvRange> argRanges) {
    assert(argRanges[0].first <= argRanges[0].second);
    assert(argRanges[1].first <= argRanges[1].second);

    APFloat min = argRanges[0].first + argRanges[1].first;
    APFloat max = argRanges[0].second + argRanges[1].second;

    return NtvRange(min, max);
  }

  template<>
  NtvRange inferMult(ArrayRef<NtvRange> argRanges) {
    // unrolled outer product
    APFloat a0b0 = argRanges[0].first * argRanges[1].first;
    APFloat a0b1 = argRanges[0].first * argRanges[1].second;
    APFloat a1b0 = argRanges[0].second * argRanges[1].first;
    APFloat a1b1 = argRanges[0].second * argRanges[1].second;

    return std::minmax({a0b0, a0b1, a1b0, a1b1});
  }
}