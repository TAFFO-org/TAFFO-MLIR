#include "Taffo/Transforms/DatatypeOptimizationPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "Taffo/Dialect/Ops.h"

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
      if (!llvm::isa<TaffoDialect>(op->getDialect())) {
        return mlir::WalkResult::advance();
      }

      DatatypeInfoAttr dtInfo =
          op->getAttr("DatatypeInfo").dyn_cast_or_null<DatatypeInfoAttr>();
      if (dtInfo == nullptr) {
        op->emitOpError() << "This op doesn't have a DatatypeInfo attribute";
        return mlir::WalkResult::interrupt();
      }
      int bitwidthDiff = targetBitwidth - dtInfo.getBitwidth();
      int newExp = dtInfo.getExponent() - bitwidthDiff;
      int newBitwidth = targetBitwidth;

      // unsure about this
      std::optional<int> newExpDiff =
          dtInfo.getExpSpan() ? std::optional<int>(std::abs(
                                    dtInfo.getExpSpan().value() + bitwidthDiff))
                              : std::nullopt;

      op->setAttr("DatatypeInfo",
                  DatatypeInfoAttr::get(op->getContext(), dtInfo.getSignd(),
                                        newExp, newBitwidth, newExpDiff));
      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace mlir