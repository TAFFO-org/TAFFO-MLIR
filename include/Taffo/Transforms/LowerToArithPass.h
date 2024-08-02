#ifndef TAFFO_TRANSFORMS_LOWERTOARITHPASS_H
#define TAFFO_TRANSFORMS_LOWERTOARITHPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::taffo
{
#define GEN_PASS_DECL_LOWERTOARITHPASS
#include "Taffo/Transforms/Passes.h.inc"

}


#endif //  TAFFO_TRANSFORMS_LOWERTOARITHPASS_H
