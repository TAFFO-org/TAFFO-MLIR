#include "Taffo/Transforms/TaffoRangeCommon.h"
#include "libaffine.hpp"
using namespace LibAffine;
using namespace ::llvm;

namespace mlir::taffo {

// unused (for now?)
template <>
NtvRange inferCast(ArrayRef<NtvRange> argRanges) {
  assert(argRanges[0].first < argRanges[0].second &&
         "Upper bound and lower bound of this range are inverted");

  return argRanges[0];
}

template <>
NtvRange inferCastToFloat(ArrayRef<NtvRange> argRanges) {
  assert(argRanges[0].first < argRanges[0].second &&
         "Upper bound and lower bound of this range are inverted");

  return argRanges[0];
}

template <>
NtvRange inferAdd(ArrayRef<NtvRange> argRanges) {
  assert(argRanges[0].first < argRanges[0].second &&
         "Upper bound and lower bound of this range are inverted");
  assert(argRanges[1].first < argRanges[1].second &&
         "Upper bound and lower bound of this range are inverted");

  APFloat min = argRanges[0].first + argRanges[1].first;
  APFloat max = argRanges[0].second + argRanges[1].second;

  return NtvRange(min, max);
}

template <>
NtvRange inferMult(ArrayRef<NtvRange> argRanges) {
  assert(argRanges[0].first < argRanges[0].second &&
         "Upper bound and lower bound of this range are inverted");
  assert(argRanges[1].first < argRanges[1].second &&
         "Upper bound and lower bound of this range are inverted");

  // unrolled outer product
  APFloat a0b0 = argRanges[0].first * argRanges[1].first;
  APFloat a0b1 = argRanges[0].first * argRanges[1].second;
  APFloat a1b0 = argRanges[0].second * argRanges[1].first;
  APFloat a1b1 = argRanges[0].second * argRanges[1].second;

  return std::minmax({a0b0, a0b1, a1b0, a1b1});
}

template <>
Var inferCast(ArrayRef<Var> argRanges) {
  assert(argRanges[0].get_range().end < argRanges[0].get_range().start &&
         "Upper bound and lower bound of this range are inverted");

  return argRanges[0];
}

template <>
Var inferCastToFloat(ArrayRef<Var> argRanges) {
  assert(argRanges[0].get_range().end < argRanges[0].get_range().start &&
         "Upper bound and lower bound of this range are inverted");

  return argRanges[0];
}

template <>
Var inferAdd(ArrayRef<Var> argRanges) {
  assert(argRanges[0].get_range().end < argRanges[0].get_range().start &&
         "Upper bound and lower bound of this range are inverted");
  assert(argRanges[1].get_range().end < argRanges[1].get_range().start &&
         "Upper bound and lower bound of this range are inverted");

  return argRanges[0] + argRanges[1];
}

template <>
Var inferMult(ArrayRef<Var> argRanges) {
  assert(argRanges[0].get_range().end < argRanges[0].get_range().start &&
         "Upper bound and lower bound of this range are inverted");
  assert(argRanges[1].get_range().end < argRanges[1].get_range().start &&
         "Upper bound and lower bound of this range are inverted");

  return argRanges[0] * argRanges[1];
}
} // namespace mlir::taffo