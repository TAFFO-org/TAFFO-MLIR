//
// Created by Paolo on 15/05/2024.
//

#ifndef TAFFO_INTERFACES_INFERTAFFORANGENTVINTERFACE_H
#define TAFFO_INTERFACES_INFERTAFFORANGENTVINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/APFloat.h"

namespace mlir::taffo {

using SetTaffoRangeFn = function_ref<void(Value, const std::pair<::llvm::APFloat,::llvm::APFloat> &)>;

} // end namespace taffo

#include "Taffo/Interfaces/InferTaffoRangeNtvInterface.h.inc"

#endif // TAFFO_INTERFACES_INFERTAFFORANGENTVINTERFACE_H
