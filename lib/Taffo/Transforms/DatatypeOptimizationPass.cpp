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
    : public mlir::taffo::impl::DatatypeOptimizationPassBase<DatatypeOptimizationPass> {
public:
  using DatatypeOptimizationPassBase::DatatypeOptimizationPassBase;

  void runOnOperation() override {}

};


} // namespace mlir