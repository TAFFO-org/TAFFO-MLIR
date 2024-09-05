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
      if (!llvm::isa<TaffoDialect>(op->getDialect()) ||
          llvm::isa<CastToFloatOp>(op)) {
        return mlir::WalkResult::advance();
      }

      RealType oldType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

      // TODO: add NaN/Inf check on exp
      int bitwidthDiff = targetBitwidth - oldType.getBitwidth();
      int newExp = oldType.getExponent() - bitwidthDiff;
      int newBitwidth = targetBitwidth;

      op->getResult(0).setType(RealType::get(
          op->getContext(), oldType.getSignd(), newExp, newBitwidth));

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
      if (!llvm::isa<TaffoDialect>(op->getDialect()) ||
          llvm::isa<CastToFloatOp>(op)) {
        return mlir::WalkResult::advance();
      }

      bool possibleOverflow =
          ::llvm::any_of(op->getUsers(), [op](Operation *addOp) {
            if (!llvm::isa<taffo::AddOp>(addOp)) {
              return false;
            }
            int childExp =
                ::llvm::dyn_cast<RealType>(addOp->getResult(0).getType())
                    .getExponent();

            int parentExp =
                ::llvm::dyn_cast<RealType>(op->getResult(0).getType())
                    .getExponent();

            return childExp == (parentExp + 1);
          });

      if (possibleOverflow) {

        RealType oldType =
            ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

        RealType newType =
            RealType::get(op->getContext(), oldType.getSignd(),
                          oldType.getExponent() + 1, oldType.getBitwidth());

        op->getResult(0).setType(newType);

        // propagate change to users
        for (Operation *childOp : op->getUsers()) {
          // currently CastToFloat this is the only op that relies on same
          // datatype because it doesn't fetch it from its operand's defining
          // op. This may be bad design, but it saves shifts down the line
          // (and can also be easily changed if the need arises in the future)
          if (llvm::isa<taffo::CastToFloatOp>(childOp))
            childOp->getResult(0).setType(newType);
          ;
        }
      }

      return mlir::WalkResult::advance();
    });

    if (result2.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace mlir