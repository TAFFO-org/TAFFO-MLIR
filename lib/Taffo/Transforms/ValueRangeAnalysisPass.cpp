#include "Taffo/Dialect/Taffo.h"
#include "Taffo/Transforms/ValueRangeAnalysisPass.h"
#include "Taffo/Transforms/NtvRangeAnalysis.h"
#include "Taffo/Dialect/Attributes.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "Taffo/Dialect/Ops.h"

namespace mlir::taffo
{
#define GEN_PASS_DEF_VALUERANGEANALYSISPASS
#include "Taffo/Transforms/Passes.h.inc"
}

using namespace ::mlir::taffo;

namespace {
  class ValueRangeAnalysisPass
      : public mlir::taffo::impl::ValueRangeAnalysisPassBase<
            ValueRangeAnalysisPass> {
  public:
    using ValueRangeAnalysisPassBase::ValueRangeAnalysisPassBase;

    void runOnOperation() override {
      mlir::Operation *op = getOperation();

      mlir::DataFlowSolver solver;

      // IntegerRangeAnalysis depends on DeadCodeAnalysis
      // Since what we are doing is very similar, we load it just in case
      // (check if it's necessary in the future)
      solver.load<mlir::dataflow::DeadCodeAnalysis>();
      solver.load<TaffoNtvRangeAnalysis>();
      if (mlir::failed(solver.initializeAndRun(op)))
        signalPassFailure();

      // only check Taffo Ops
      if (!llvm::isa<TaffoDialect>(op->getDialect())) {
        return;
      }

      const TaffoRangeLattice *opRange =
          solver.lookupState<TaffoRangeLattice>(
              op->getResult(0));
      if (!opRange || opRange->getValue().isUninitialized()) {
        op->emitOpError()
            << "Found op without a set range; have all variables"
               "been assigned a range?";
        signalPassFailure();
      }
      NtvRange range = opRange->getValue().getValue();
      bool signd = range.first.isNegative() || range.second.isNegative();

      // Hardcoding for f32, in the future it will need to work off of either
      // precision, number of significant digits, or a global parameter
      //const int maxBitwidth = 32;
      const int maxSignificantDigits = 24;

      // we expect the exponent to be small (<2^31), this might
      // need to be changed for arbitrary precision scientific
      // computing
      int lf = range.first.getExactLog2Abs();
      int ls = range.second.getExactLog2Abs();
      int max_exp = std::max(lf, ls);

      // temporary hack, is it good enough?
      int bitwidth = maxSignificantDigits;

      int exponent = max_exp - bitwidth;
      op->setAttr("DatatypeInfo",
          DatatypeInfoAttr::get(op->getContext(), signd, exponent, bitwidth));

      return;
    }
  };
}