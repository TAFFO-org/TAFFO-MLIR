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

namespace {
  class ValueRangeAnalysisPass
      : public mlir::taffo::impl::ValueRangeAnalysisPassBase<
            ValueRangeAnalysisPass> {
  public:
    using ValueRangeAnalysisPassBase::ValueRangeAnalysisPassBase;

    void runOnOperation() override {}
  };
}