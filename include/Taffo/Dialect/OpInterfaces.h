//
// Created by Paolo on 15/05/2024.
//

#ifndef TAFFO_INTERFACES_INFERTAFFORANGEINTERFACE_H
#define TAFFO_INTERFACES_INFERTAFFORANGEINTERFACE_H

#include "libaffine.hpp"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/APFloat.h"

namespace mlir::taffo {

using SetTaffoRangeFn = function_ref<void(
    Value, const std::pair<::llvm::APFloat, ::llvm::APFloat> &)>;

using SetTaffoAffineRangeFn = function_ref<void(Value, const LibAffine::Var &)>;

} // namespace mlir::taffo

#include "Taffo/Dialect/OpInterfaces.h.inc"

#endif // TAFFO_INTERFACES_INFERTAFFORANGENTVINTERFACE_H
