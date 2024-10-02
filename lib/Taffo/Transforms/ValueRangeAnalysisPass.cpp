#include "Taffo/Transforms/ValueRangeAnalysisPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
#include "Taffo/Transforms/NtvRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "Taffo/Dialect/Ops.h"

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
    if (mlir::failed(solver.initializeAndRun(module)))
      signalPassFailure();

    auto result = module->walk([&](mlir::Operation *op) {
      if (!llvm::isa<TaffoDialect>(op->getDialect()) ||
          llvm::isa<CastToFloatOp>(op)) {
        return mlir::WalkResult::advance();
      }
      const TaffoRangeLattice *opRange =
          solver.lookupState<TaffoRangeLattice>(op->getResult(0));
      if (!opRange || opRange->getValue().isUninitialized()) {
        op->emitOpError() << "Found op without a set range; have all variables"
                             " been assigned a range?";
        return mlir::WalkResult::interrupt();
      }
      NtvRange range = opRange->getValue().getValue();
      bool signd = range.first.isNegative() || range.second.isNegative();

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
      int lf = getLog2(range.first);
      int ls = getLog2(range.second);

      int max_exp = std::max(lf, ls);

      // since we use 2's complement for fixed point numbers, we
      // need to account for the fact that the MSB will have negative
      // weight
      max_exp = signd ? max_exp + 1 : max_exp;

      RealType type = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());
      int bitwidth = type.getBitwidth() ? type.getBitwidth() : maxSignificantDigits;

      // the left-most bit of an  integer has 2^(bitwidth-1) weight,
      // we want the leftmost bit of the fixed-point integer to have
      // weight 2^max_exp, hence the "-1"
      int exponent = max_exp - (bitwidth - 1);

      // if one or more of my operands are signed, I am also signed
      if (!signd && !llvm::isa<CastToRealOp>(op) &&
          llvm::any_of(op->getOperands(), [op](mlir::Value v) {
            return ::llvm::dyn_cast<RealType>(v.getType()).getSignd();
          })) {
        signd = true;
        exponent += 1;
      }

      op->getResult(0).setType(
          RealType::get(op->getContext(), signd, exponent, bitwidth));

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace