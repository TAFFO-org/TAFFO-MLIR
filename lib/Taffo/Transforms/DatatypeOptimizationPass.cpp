#include "Taffo/Transforms/DatatypeOptimizationPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "Taffo/Dialect/Ops.h"
#include <iostream>
namespace mlir::taffo {
#define GEN_PASS_DEF_DATATYPEOPTIMIZATIONPASS
#include "Taffo/Transforms/Passes.h.inc"
} // namespace mlir::taffo

using namespace ::mlir::taffo;

namespace mlir {
class DatatypeOptimizationPass
    : public mlir::taffo::impl::DatatypeOptimizationPassBase<
          DatatypeOptimizationPass> {
public:
  using DatatypeOptimizationPassBase::DatatypeOptimizationPassBase;

  void runOnOperation() override {
    mlir::Operation *module = getOperation();

    // this will become a parameter captured from pass invocation at some point
    const int targetBitwidth = 32;

    auto result = module->walk([&](mlir::Operation *op) {
      if (llvm::none_of(op->getResultTypes(),
                        [](Type t) { return llvm::isa<RealType>(t); })) {
        return mlir::WalkResult::advance();
      }

      auto adaptToBitwidth = [targetBitwidth](Value v) {
        RealType oldType = ::llvm::dyn_cast<RealType>(v.getType());

        // TODO: add NaN/Inf check on exp
        int bitwidthDiff = targetBitwidth - oldType.getBitwidth();
        int newExp = oldType.getExponent() - bitwidthDiff;
        int newBitwidth = targetBitwidth;

        v.setType(RealType::get(v.getContext(), oldType.getSignd(), newExp,
                                newBitwidth));
      };

      auto loop = llvm::dyn_cast<mlir::LoopLikeOpInterface>(op);
      if (loop) {
        adaptToBitwidth(loop.getRegionIterArgs().front());
        adaptToBitwidth(loop.getInits().front());
        adaptToBitwidth(loop->getResults().front());
      } else {
        adaptToBitwidth(op->getResult(0));
      }

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();

  }
};
} // namespace mlir