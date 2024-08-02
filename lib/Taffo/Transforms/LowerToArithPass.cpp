#include "Taffo/Dialect/Taffo.h"
#include "Taffo/Transforms/LowerToArithPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "Taffo/Dialect/Ops.h"

namespace mlir::taffo
{
#define GEN_PASS_DEF_LOWERTOARITHPASS
#include "Taffo/Transforms/Passes.h.inc"
}

using namespace ::mlir::taffo;

namespace {
class LowerToArithPass
    : public mlir::taffo::impl::LowerToArithPassBase<
          LowerToArithPass> {
public:
  using LowerToArithPassBase::LowerToArithPassBase;

  void runOnOperation() override {
    mlir::Operation *module = getOperation();

  }
};
}