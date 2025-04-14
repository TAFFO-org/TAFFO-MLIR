#ifndef TAFFO_TRANSFORMS_RAISETOTAFFO_PASS_H
#define TAFFO_TRANSFORMS_RAISETOTAFFO_PASS_H

#define DEBUG_TYPE "raise-to-taffo"

#include "mlir/Pass/Pass.h"
namespace mlir::taffo {
#define GEN_PASS_DECL_RAISETOTAFFOPASS
#include "Taffo/Transforms/Passes.h.inc"
} // namespace mlir::taffo

#endif // TAFFO_TRANSFORMS_RAISETOTAFFO_PASS_H