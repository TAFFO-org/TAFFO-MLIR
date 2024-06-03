#include "Taffo/ValueRangeAnalysisPass.h"
#include "Taffo/TaffoDialect.h"
#include "Taffo/TaffoNtvRangeAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir::taffo
{
#define GEN_PASS_DEF_VALUERANGEANALYSISPASS
#include "Taffo/Passes.h.inc"
}

namespace {
  class ValueRangeAnalysisPass
      : public mlir::taffo::impl::ValueRangeAnalysisPassBase<
            ValueRangeAnalysisPass> {
  public:
    using ValueRangeAnalysisPassBase::ValueRangeAnalysisPassBase;

    void runOnOperation() override {}
  }
}