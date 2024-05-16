//
// Created by Paolo on 15/05/2024.
//

#ifndef TAFFO_INFERTAFFORANGENTVINTERFACE_H
#define TAFFO_INFERTAFFORANGENTVINTERFACE_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::taffo {

using SetTaffoRangeFn = function_ref<void(Value, const std::pair<::llvm::APFloat,::llvm::APFloat> &)>;

} // end namespace taffo

#include "Taffo/InferTaffoRangeNtvInterface.h.inc"

#endif // TAFFO_INFERTAFFORANGENTVINTERFACE_H
