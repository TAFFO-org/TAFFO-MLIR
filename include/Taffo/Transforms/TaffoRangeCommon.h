#ifndef TAFFO_TRANSFORMS_TAFFORANGECOMMON_H
#define TAFFO_TRANSFORMS_TAFFORANGECOMMON_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APFloat.h"

namespace mlir::taffo {

// range type for intervarl arithmetic
using NtvRange = std::pair<::llvm::APFloat, ::llvm::APFloat>;

// using a template to support both affine and interval arithmetic
template <typename T>
T inferCast(llvm::ArrayRef<T> argRanges);

template <typename T>
T inferCastToFloat(llvm::ArrayRef<T> argRanges);

template <typename T>
T inferAdd(llvm::ArrayRef<T> argRanges);

template <typename T>
T inferMult(llvm::ArrayRef<T> argRanges);

} // namespace taffo

#endif // TAFFO_TRANSFORMS_TAFFORANGECOMMON_H
