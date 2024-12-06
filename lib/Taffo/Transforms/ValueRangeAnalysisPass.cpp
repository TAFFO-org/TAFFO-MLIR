#include "Taffo/Transforms/ValueRangeAnalysisPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
#include "Taffo/Transforms/NtvRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "Taffo/Dialect/Ops.h"
#include "Taffo/Transforms/AffineRangeAnalysis.hpp"
namespace mlir::taffo {
#define GEN_PASS_DEF_VALUERANGEANALYSISPASS
#include "Taffo/Transforms/Passes.h.inc"
} // namespace mlir::taffo

using namespace ::mlir::taffo;

namespace {
class ValueRangeAnalysisPass
    : public mlir::taffo::impl::ValueRangeAnalysisPassBase<
          ValueRangeAnalysisPass> {
public:
  using ValueRangeAnalysisPassBase::ValueRangeAnalysisPassBase;

  void runOnOperation() override {
    mlir::Operation *module = getOperation();

    mlir::DataFlowSolver solver;

    // IntegerRangeAnalysis depends on DeadCodeAnalysis
    // Since what we are doing is very similar, we load it just in case
    // (check if it's necessary in the future)
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<TaffoNtvRangeAnalysis>();
    solver.load<TaffoAffineRangeAnalysis>();

    if (mlir::failed(solver.initializeAndRun(module)))
      signalPassFailure();

    auto result = module->walk([&](mlir::Operation *op) {
      // LLVM_DEBUG(llvm::dbgs() << "Visiting op with type: "
      //                         << op->getResultTypes() << *op << "\n");
      if (llvm::isa<mlir::scf::YieldOp>(op)) {
        handleYield(op);
        return mlir::WalkResult::advance();
      }

      if (!llvm::isa<TaffoDialect>(op->getDialect()) ||
          llvm::isa<CastToFloatOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping op: " << *op << "\n");
        return mlir::WalkResult::advance();
      }
      // Lookup the range of the operation in Ntv Lattice
      const TaffoRangeLattice *opNtvRange =
          solver.lookupState<TaffoRangeLattice>(op->getResult(0));
      if (!opNtvRange || opNtvRange->getValue().isUninitialized()) {
        op->emitOpError() << "Found op without a set range; have all variables"
                             " been assigned a range?";
        return mlir::WalkResult::interrupt();
      }

      // Lookup the range of the operation in Affine Lattice
      const TaffoAffineRangeLattice *opAffineRange =
          solver.lookupState<TaffoAffineRangeLattice>(op->getResult(0));
      if (!opAffineRange || opAffineRange->getValue().isUninitialized()) {
        op->emitOpError() << "Found op without a set range; have all variables"
                             "been assigned a range?";
        return mlir::WalkResult::interrupt();
      }

      // Select the smallets range of the two based on their radius
      TaffoValueRange::NtvRange ntvRange = opNtvRange->getValue().getValue();
      auto affineRange = opAffineRange->getValue().getValue().get_range();
      std::unique_ptr<LibAffine::Range> final_range =
          std::make_unique<LibAffine::Range>(ntvRange.first, ntvRange.second);

      bool signd =
          final_range->start.isNegative() || final_range->end.isNegative();

      // Hardcoding for f32, in the future it will need to work off of either
      // precision, number of significant digits, or a global parameter
      // const int maxBitwidth = 32;
      const int maxSignificantDigits = 24;

      auto getLog2 = [](::llvm::APFloat f) -> int {
        return static_cast<int>(
            std::floor(std::log2(std::abs(f.convertToDouble()))));
      };

      // we expect the exponent to be small (<2^31), this might
      // need to be changed for arbitrary precision scientific
      // computing (that being said, quad precision exponent is
      // 15 bits wide......)
      int lf = getLog2(final_range->start);
      int ls = getLog2(final_range->end);

      int max_exp = std::max(lf, ls);

      // since we use 2's complement for fixed point numbers, we
      // need to account for the fact that the MSB will have negative
      // weight
      max_exp = signd ? max_exp + 1 : max_exp;

      RealType type = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());
      int bitwidth =
          type.getBitwidth() ? type.getBitwidth() : maxSignificantDigits;

      // the left-most bit of an  integer has 2^(bitwidth-1) weight,
      // we want the leftmost bit of the fixed-point integer to have
      // weight 2^max_exp, hence the "-1"
      int exponent = max_exp - (bitwidth - 1);

      // if one or more of my operands are signed, I am also signed
      if (!signd && !llvm::isa<CastToRealOp>(op) &&
          llvm::any_of(op->getOperands(), [](mlir::Value v) {
            return ::llvm::dyn_cast<RealType>(v.getType()).getSignd();
          })) {
        signd = true;
        exponent += 1;
      }

      auto new_type =
          RealType::get(op->getContext(), signd, exponent, bitwidth);
      op->getResult(0).setType(new_type);

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }

  void handleYield(mlir::Operation *op) {
    auto getMSB = [](RealType t) { return t.getBitwidth() + t.getExponent(); };

    auto results = op->getOperands();
    for (auto res : results) {
      LLVM_DEBUG(llvm::dbgs() << "operands: " << res << "\n");
    }
    if (llvm::range_size(results) == 0) {
      return;
    }

    mlir::LoopLikeOpInterface parent =
        llvm::dyn_cast<mlir::LoopLikeOpInterface>(op->getParentOp());
    if (!parent) {
      return;
    }

    // propagate backwards
    auto inits = parent.getInits();
    for (auto it : llvm::zip(inits, results)) {
      mlir::Value init = std::get<0>(it);
      mlir::Value result = std::get<1>(it);
      RealType resType = llvm::dyn_cast<RealType>(result.getType());
      RealType initType = init.getType().cast<RealType>();
      if (getMSB(resType) > getMSB(initType)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Aligning " << init << " to " << resType << "\n");
        if (llvm::range_size(init.getUsers()) > 1) {
          mlir::OpBuilder b = mlir::OpBuilder(parent, nullptr);
          auto align =
              b.create<mlir::taffo::AlignOp>(parent.getLoc(), resType, init);
          init.replaceUsesWithIf(
              align.getResult(),
              [parent](mlir::OpOperand &U) { return U.getOwner() == parent; });
        } else {
          init.setType(resType);
        }
      }

      // Align yield operands if their range is narrower than the intial values
      else if (getMSB(resType) < getMSB(initType)) {
        if (llvm::range_size(init.getUsers()) > 1) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Aligning " << init << " to " << resType << "\n");
          mlir::OpBuilder b = mlir::OpBuilder(op, nullptr);
          auto align =
              b.create<mlir::taffo::AlignOp>(op->getLoc(), initType, result);
          result.replaceUsesWithIf(align.getResult(), [op](mlir::OpOperand &U) {
            return U.getOwner() == op;
          });
        } else {
          result.setType(initType);
        }
      }
    }

    // propagate forward
    auto iterArgs = parent.getRegionIterArgs();
    for (auto it : llvm::zip(iterArgs, results)) {
      mlir::Value result = std::get<1>(it);
      mlir::BlockArgument iterArg = std::get<0>(it);
      RealType resType = llvm::dyn_cast<RealType>(result.getType());

      LLVM_DEBUG(llvm::dbgs() << "setting iterArg type " << std::get<0>(it)
                              << "to: " << resType << "\n");

      iterArg.setType(resType);
    }

    auto parentResults = parent->getResults();
    for (auto it : llvm::zip(parentResults, results)) {
      mlir::Value result = std::get<1>(it);
      mlir::Value parentResult = std::get<0>(it);
      RealType resType = llvm::dyn_cast<RealType>(result.getType());

      LLVM_DEBUG(llvm::dbgs() << "setting parent type " << std::get<0>(it)
                              << "to: " << resType << "\n");
      parentResult.setType(resType);
    }
  }
};

} // namespace