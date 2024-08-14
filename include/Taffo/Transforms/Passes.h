#ifndef TAFFO_TRANSFORMS_PASSES_H
#define TAFFO_TRANSFORMS_PASSES_H

#include "Taffo/Transforms/ValueRangeAnalysisPass.h"
#include "Taffo/Transforms/LowerToArithPass.h"
#include "Taffo/Transforms/DatatypeOptimizationPass.h"

namespace mlir::taffo
{
#define GEN_PASS_REGISTRATION
#include "Taffo/Transforms/Passes.h.inc"
}

#endif // TAFFO_TRANSFORMS_PASSES_H
