#ifndef TAFFO_TRANSFORMS_NTVRANGEANALYSIS_H
#define TAFFO_TRANSFORMS_NTVRANGEANALYSIS_H

#include "Taffo/Dialect/OpInterfaces.h"
#include "Taffo/Transforms/TaffoRangeCommon.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace mlir {
namespace taffo {

class TaffoValueRange {
public:
  using NtvRange = mlir::taffo::NtvRange;

  static TaffoValueRange getMaxRange(Value value);

  TaffoValueRange(std::optional<NtvRange> value = std::nullopt)
      : value(std::move(value)) {}

  /// Whether the range is uninitialized. This happens when the state hasn't
  /// been set during the analysis.
  bool isUninitialized() const { return !value.has_value(); }

  /// Get the known integer value range.
  const NtvRange &getValue() const {
    assert(!isUninitialized());
    return *value;
  }

  /// Compare two ranges.
  bool operator==(const TaffoValueRange &rhs) const {
    return value == rhs.value;
  }

  /// Take the union of two ranges.
  static TaffoValueRange join(const TaffoValueRange &lhs,
                              const TaffoValueRange &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;

    NtvRange lhs_range = lhs.getValue();
    NtvRange rhs_range = rhs.getValue();
    return TaffoValueRange{std::minmax({lhs_range.first, lhs_range.second,
                                        rhs_range.first, rhs_range.second})};
  }

  /// Print the value range.
  void print(raw_ostream &os) const {
    os << "taffo range: [" << getValue().first.convertToDouble() << ", "
       << getValue().second.convertToDouble() << "]";
  }

private:
  /// The known integer value range.
  std::optional<NtvRange> value;
};

class TaffoRangeLattice : public dataflow::Lattice<TaffoValueRange> {
public:
  using Lattice::Lattice;

  /// If the range can be narrowed to a constant, update the constant
  /// value of the SSA value.
  void onUpdate(DataFlowSolver *solver) const override;
};

class TaffoNtvRangeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TaffoRangeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// At an entry point, we cannot reason about value ranges.
  void setToEntryState(TaffoRangeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(TaffoValueRange::getMaxRange(
                                    lattice->getAnchor())));
  }

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferTaffoRangeNtvInterface`.
  mlir::LogicalResult visitOperation(Operation *op,
                      ArrayRef<const TaffoRangeLattice *> operands,
                      ArrayRef<TaffoRangeLattice *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow. This
  /// function calls `InferIntRangeInterface` to provide values for block
  /// arguments or tries to reduce the range on loop induction variables with
  /// known bounds.
  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ArrayRef<TaffoRangeLattice *> argLattices,
                                    unsigned firstIndex) override;

private:
  bool hitTripCount (Value v);
  std::map<Operation*, int64_t> loops;
  std::map<Operation*, int64_t> opVisits;
};

} // end namespace taffo
} // end namespace mlir

#endif // TAFFO_TRANSFORMS_NTVRANGEANALYSIS_H
