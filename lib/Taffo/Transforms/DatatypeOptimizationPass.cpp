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
      std::optional<int> newExpSpan =
          dtInfo.getExpSpan() ? std::optional<int>(std::abs(
                                    dtInfo.getExpSpan().value() + bitwidthDiff))
                              : std::nullopt;

      op->setAttr("DatatypeInfo",
                  DatatypeInfoAttr::get(op->getContext(), dtInfo.getSignd(),
                                        newExp, newBitwidth, newExpSpan));

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();

    // This manipulation on addOp in necessary to prevent overflow. This can
    // be achieved in different parts of the pipeline if the need arises in
    // the future. It relies on the fact that all taffo ops are converting to
    // the same bitwidth, as this has relevance wrt to exponent semantics and
    // fixed-point alignment
    auto result2 = module->walk([&](mlir::Operation *op) {
      if (!llvm::isa<TaffoDialect>(op->getDialect())) {
        return mlir::WalkResult::advance();
      }

      bool possibleOverflow =
          ::llvm::any_of(op->getUsers(), [op](Operation *addOp) {
            if (!llvm::isa<taffo::AddOp>(addOp)) {
              return false;
            }
            DatatypeInfoAttr childDt =
                addOp->getAttr("DatatypeInfo").dyn_cast<DatatypeInfoAttr>();
            DatatypeInfoAttr parentDt =
                op->getAttr("DatatypeInfo").dyn_cast<DatatypeInfoAttr>();

            return childDt.getExponent() == (parentDt.getExponent() + 1);
          });

      if (possibleOverflow) {

        DatatypeInfoAttr oldDt =
            op->getAttr("DatatypeInfo").dyn_cast<DatatypeInfoAttr>();

        DatatypeInfoAttr newDt = DatatypeInfoAttr::get(
            op->getContext(), oldDt.getSignd(), oldDt.getExponent() + 1,
            oldDt.getBitwidth(),
            oldDt.getExpSpan()
                ? std::optional<int>(oldDt.getExpSpan().value() + 1)
                : std::nullopt);

        op->setAttr("DatatypeInfo", newDt);

        // propagate change to users
        for (Operation *childOp : op->getUsers()) {
          // currently CastToFloat this is the only op that relies on same
          // datatype because it doesn't fetch it from its operand's defining
          // op. This may be bad design, but it saves shifts down the line
          // (and can also be easily changed if the need arises in the future)
          if (llvm::isa<taffo::CastToFloatOp>(childOp))
            childOp->setAttr("DatatypeInfo", newDt);
        }
      }

      return mlir::WalkResult::advance();
    });

    if (result2.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace mlir