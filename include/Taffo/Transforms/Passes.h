#ifndef TAFFO_TRANSFORMS_PASSES_H
#define TAFFO_TRANSFORMS_PASSES_H

#include "Taffo/Transforms/DatatypeInitialization.h"
#include "Taffo/Transforms/TaffoToArith.h"
#include "Taffo/Transforms/DatatypeOptimization.h"

namespace mlir::taffo
{
#define GEN_PASS_REGISTRATION
#include "Taffo/Transforms/Passes.h.inc"
}

#endif // TAFFO_TRANSFORMS_PASSES_H
