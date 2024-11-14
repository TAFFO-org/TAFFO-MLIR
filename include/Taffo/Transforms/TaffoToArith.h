#ifndef TAFFO_TRANSFORMS_TAFFOTOARITH_H
#define TAFFO_TRANSFORMS_TAFFOTOARITH_H

#include "mlir/Pass/Pass.h"

namespace mlir::taffo
{
#define GEN_PASS_DECL_TAFFOTOARITHCONVERSIONPASS
#include "Taffo/Transforms/Passes.h.inc"

}


#endif //  TAFFO_TRANSFORMS_TAFFOTOARITH_H
