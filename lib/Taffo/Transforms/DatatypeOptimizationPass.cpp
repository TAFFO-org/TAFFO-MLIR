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
private:
  const int targetBitwidth = 32;

public:
  using DatatypeOptimizationPassBase::DatatypeOptimizationPassBase;

  void adaptToBitwidth(Value v) {
    RealType oldType = ::llvm::dyn_cast<RealType>(v.getType());

    // TODO: add NaN/Inf check on exp
    int bitwidthDiff = targetBitwidth - oldType.getBitwidth();
    int newExp = oldType.getExponent() - bitwidthDiff;
    int newBitwidth = targetBitwidth;

    v.setType(
        RealType::get(v.getContext(), oldType.getSignd(), newExp, newBitwidth));
  }

  void runOnOperation() override {
    mlir::Operation *module = getOperation();

    // this will become a parameter captured from pass invocation at some point

    auto result = module->walk([&](mlir::Operation *op) {
      if (llvm::none_of(op->getResultTypes(),
                        [](Type t) { return llvm::isa<RealType>(t); })) {
        return mlir::WalkResult::advance();
      }

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

    // This manipulation on users addOp is necessary to prevent overflow. This
    // can be achieved in different parts of the pipeline if the need arises in
    // the future. It relies on the fact that all taffo ops are converting to
    // the same bitwidth, as this has relevance wrt to exponent semantics and
    // fixed-point alignment
    auto result2 = module->walk([&](mlir::Operation *op) {
      if (llvm::none_of(op->getResultTypes(),
                        [](Type t) { return llvm::isa<RealType>(t); })) {
        return mlir::WalkResult::advance();
      }

      auto possibleOverflow = [op](Operation *addOp) {
        if (!llvm::isa<taffo::AddOp>(addOp)) {
          return false;
        }

        auto getExp = [](Value v) {
          return ::llvm::dyn_cast<RealType>(v.getType()).getExponent();
        };

        SmallVector<int> operandExps;
        for (auto opr : addOp->getOperands()) {
          operandExps.push_back(getExp(opr));
        }

        int maxOperandExp =
            *std::max_element(operandExps.begin(), operandExps.end());

        int resExp = getExp(addOp->getResult(0));

        return resExp == (maxOperandExp + 1) &&
               getExp(op->getResult(0)) == maxOperandExp;
      };

      auto usersWithOF =
          llvm::make_filter_range(op->getUsers(), possibleOverflow);

      if (usersWithOF.empty()) {
        return mlir::WalkResult::advance();
      }

      RealType oldType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

      RealType newType =
          RealType::get(op->getContext(), oldType.getSignd(),
                        oldType.getExponent() + 1, oldType.getBitwidth());
      mlir::OpBuilder b = mlir::OpBuilder(op, nullptr);
      b.setInsertionPointAfter(op);
      auto align = b.create<mlir::taffo::AlignOp>(op->getLoc(), newType,
                                                  op->getResult(0));

      op->getResult(0).replaceUsesWithIf(
          align.getResult(), [usersWithOF](mlir::OpOperand &U) {
            return llvm::any_of(usersWithOF, [&](Operation *user) {
              return user == U.getOwner();
            });
          });

      return mlir::WalkResult::advance();
    });

    if (result2.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace mlir