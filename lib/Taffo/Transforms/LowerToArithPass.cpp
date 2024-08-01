#include "Taffo/Dialect/Taffo.h"
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


    auto result = module->walk([&](mlir::Operation *op) {
      if (!llvm::isa<TaffoDialect>(op->getDialect())) {
        return mlir::WalkResult::advance();
      }
      const TaffoRangeLattice *opRange =
          solver.lookupState<TaffoRangeLattice>(
              op->getResult(0));
      if (!opRange || opRange->getValue().isUninitialized()) {
        op->emitOpError()
            << "Found op without a set range; have all variables"
               "been assigned a range?";
        return mlir::WalkResult::interrupt();
      }

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};
}