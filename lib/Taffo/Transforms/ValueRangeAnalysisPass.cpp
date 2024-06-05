#include "Taffo/Dialect/TaffoDialect.h"
#include "Taffo/Transforms/ValueRangeAnalysisPass.h"
#include "Taffo/Transforms/NtvRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

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
        if (!llvm::isa<AddOp, AssignOp>(*op)) {
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