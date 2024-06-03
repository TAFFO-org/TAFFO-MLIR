#ifndef TAFFO_VALUERANGEANALYSISPASS_H
#define TAFFO_VALUERANGEANALYSISPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::taffo
{
#define GEN_PASS_DECL_VALUERANGEANALYSISPASS
#include "Taffo/Passes.h.inc"

}


#endif //  TAFFO_VALUERANGEANALYSISPASS_H
