#ifndef TAFFO_PASSES_H
#define TAFFO_PASSES_H

#include "Taffo/ValueRangeAnalysisPass.h"

namespace mlir::taffo
{
#define GEN_PASS_REGISTRATION
#include "Taffo/Passes.h.inc"
}

#endif // TAFFO_PASSES_H
