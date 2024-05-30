#ifndef TAFFO_TAFFORANGECOMMON_H
#define TAFFO_TAFFORANGECOMMON_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APFloat.h"

namespace taffo {

// range type for intervarl arithmetic
using NtvRange = std::pair<::llvm::APFloat, ::llvm::APFloat>;

// using a template to support both affine and interval arithmetic
template <typename T>
T inferAssign(llvm::ArrayRef<T> argRanges);

template <typename T>
T inferAdd(llvm::ArrayRef<T> argRanges);

template <typename T>
T inferMult(llvm::ArrayRef<T> argRanges);

} // namespace taffo

#endif // TAFFO_TAFFORANGECOMMON_H
