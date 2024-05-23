//
// Created by Paolo on 22/05/2024.
//

#ifndef TAFFO_NTVRANGEANALYSIS_H
#define TAFFO_NTVRANGEANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "Taffo/InferTaffoRangeNtvInterface.h"
#include "Taffo/TaffoRangeCommon.h"

namespace mlir {
namespace dataflow {

class TaffoValueRange {
public:

  static NtvRange getMaxRange(Value value);

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
  bool operator==(const NtvRange &rhs) const {
    return value == rhs.value;
  }

  /// Take the union of two ranges.
  static TaffoValueRange join(const NtvRange &lhs,
                              const NtvRange &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    return std::minmax({lhs.first, lhs.second, rhs.first, rhs.second})};
  }

  /// Print the value range.
  void print(raw_ostream &os) const { os << value; }

private:
  /// The known integer value range.
  std::optional<NtvRange> value;
};



class TaffoRangeLattice : public Lattice<TaffoValueRange> {
public:
  using Lattice::Lattice;

  /// If the range can be narrowed to a constant, update the constant
  /// value of the SSA value.
  void onUpdate(DataFlowSolver *solver) const override;
};


class TaffoNtvRangeAnalysis
    : public SparseForwardDataFlowAnalysis<TaffoRangeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// At an entry point, we cannot reason about value ranges.
  void setToEntryState(IntegerValueRangeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(TaffoValueRange::getMaxRange(
                                    lattice->getPoint())));
  }

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferTaffoRangeNtvInterface`.
  void visitOperation(Operation *op,
                      ArrayRef<const TaffoRangeLattice *> operands,
                      ArrayRef<TaffoRangeLattice *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow. This
  /// function calls `InferIntRangeInterface` to provide values for block
  /// arguments or tries to reduce the range on loop induction variables with
  /// known bounds.
  void
  visitNonControlFlowArguments(Operation *op, const RegionSuccessor &successor,
                               ArrayRef<TaffoRangeLattice *> argLattices,
                               unsigned firstIndex) override;
};


} // end namespace dataflow
} // end namespace mlir

#endif // TAFFO_NTVRANGEANALYSIS_H
