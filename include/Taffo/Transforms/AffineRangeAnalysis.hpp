#ifndef TAFFO_TRANSFORMS_AFFINERANGEANALYSIS_H
#define TAFFO_TRANSFORMS_AFFINERANGEANALYSIS_H

#include "libaffine.hpp"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "value-range-analysis"

namespace mlir {
namespace taffo {
class TaffoAffineValueRange {
public:
  static TaffoAffineValueRange getMaxRange(Value value);

  TaffoAffineValueRange(std::optional<LibAffine::Var> val = std::nullopt)
      : value(std::move(val)) {}

  /// Whether the range is uninitialized. This happens when the state hasn't
  /// been set during the analysis.
  bool isUninitialized() const { return !value.has_value(); }

  /// Get the known integer value range.
  const LibAffine::Var &getValue() const {
    assert(!isUninitialized());
    return *value;
  }

  /// Compare two ranges.
  bool operator==(const TaffoAffineValueRange &rhs) const {
    return value == rhs.value;
    // return true;
  }

  /// Take the union of two ranges.
  static TaffoAffineValueRange join(const TaffoAffineValueRange &lhs,
                                    const TaffoAffineValueRange &rhs) {
    if (lhs.isUninitialized())
      return rhs;

    if (rhs.isUninitialized())
      return lhs;
    // LLVM_DEBUG(llvm::dbgs()
    //            << "Joining ranges: " << lhs.getValue().print_affine_form()
    //            << " and " << rhs.getValue().print_affine_form() << "\n");
    // LLVM_DEBUG(
    //     llvm::dbgs()
    //     << "Joined range: "
    //     << lhs.getValue().join(rhs.getValue()).print_affine_form() << "\n["
    //     << lhs.getValue()
    //            .join(rhs.getValue())
    //            .get_range()
    //            .start.convertToDouble()
    //     << ","
    //     <<
    //     lhs.getValue().join(rhs.getValue()).get_range().end.convertToDouble()
    //     << "]\n");
    return TaffoAffineValueRange{lhs.getValue().join(rhs.getValue())};
  }

  /// Print the value range.
  void print(raw_ostream &os) const {
    auto range = getValue();
    os << getValue().print();
  }

private:
  /// The known integer value range.
  std::optional<LibAffine::Var> value;
};

class TaffoAffineRangeLattice
    : public dataflow::Lattice<TaffoAffineValueRange> {
public:
  using Lattice::Lattice;

  /// If the range can be narrowed to a constant, update the constant
  /// value of the SSA value.
  void onUpdate(DataFlowSolver *solver) const override;
};

class TaffoAffineRangeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TaffoAffineRangeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// At an entry point, we cannot reason about value ranges.
  void setToEntryState(TaffoAffineRangeLattice *lattice) override {
    propagateIfChanged(
        lattice,
        lattice->join(TaffoAffineValueRange::getMaxRange(lattice->getPoint())));
  }

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferTaffoRangeInterface`.
  void visitOperation(Operation *op,
                      ArrayRef<const TaffoAffineRangeLattice *> operands,
                      ArrayRef<TaffoAffineRangeLattice *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow. This
  /// function calls `InferIntRangeInterface` to provide values for block
  /// arguments or tries to reduce the range on loop induction variables with
  /// known bounds.
  void
  visitNonControlFlowArguments(Operation *op, const RegionSuccessor &successor,
                               ArrayRef<TaffoAffineRangeLattice *> argLattices,
                               unsigned firstIndex) override;

private:
  bool hitTripCount(Value v);
  std::map<Operation *, int64_t> loops;
  std::map<Operation *, int64_t> opVisits;
};
} // namespace taffo
} // namespace mlir

#endif // TAFFO_TRANSFORMS_AFFINERANGEANALYSIS_H