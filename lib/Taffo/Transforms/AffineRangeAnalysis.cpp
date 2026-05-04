#include "Taffo/Transforms/AffineRangeAnalysis.hpp"
// most includes copied from mlir/Analysis/DataFlow/IntegerRangeAnalysis.cpp
// might want to clean up later
#include "Taffo/Dialect/Ops.h"
#include "Taffo/Dialect/Taffo.h"
#include "Taffo/Dialect/OpInterfaces.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <llvm/ADT/APFloat.h>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::taffo;
using namespace LibAffine;
TaffoAffineValueRange TaffoAffineValueRange::getMaxRange(Value value) {

  // placeholder, needs to work based on annotations of Value in the future
  // instead of defaulting to APFloat::IEEEdouble()
  // should this return [-max, max] instead?
  return TaffoAffineValueRange(
      Var(LibAffine::Range(APFloat::getInf(APFloat::IEEEdouble(), true),
                           APFloat::getInf(APFloat::IEEEdouble(), false))));
}

static int64_t estimateTripCount(Operation *op) {
  const int64_t defaultTripCount = 100;

  auto forOp = dyn_cast<scf::ForOp>(op);

  if (forOp) {
    std::optional<int64_t> tripCount = constantTripCount(
        forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep());
    if (tripCount)
      return tripCount.value();
  }

  op->emitWarning(
      "Variable loop trip count detected, falling back to default trip count");
  return defaultTripCount;
}

bool TaffoAffineRangeAnalysis::hitTripCount(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (arg)
    return true;

  auto defOp = v.getDefiningOp();
  if (!defOp) {
    return true;
  }

  auto parent = defOp->getParentOp();
  if (!parent) {
    return true;
  }

  auto searchParent = this->loops.find(parent);
  // if parent is not in loop map there's no need to track visits
  if (searchParent == this->loops.end()) {
    return true;
  }

  auto searchOp = this->opVisits.find(defOp);
  // if the op is not in opVisits, it doesn't have a trip count
  if (searchOp == this->opVisits.end()) {
    return true;
  }

  return searchOp->second >= searchParent->second;
}

void TaffoAffineRangeLattice::onUpdate(DataFlowSolver *solver) const {
  Lattice::onUpdate(solver);

  // If the range can be narrowed to a constant, update the constant
  // value of the SSA value.
  const auto range = getValue().getValue().get_range();
  std::optional<APFloat> constant = range.start == range.end
                                        ? std::optional<APFloat>(range.start)
                                        : std::nullopt;

  auto value = point.get<Value>();
  auto *cv = solver->getOrCreateState<Lattice<ConstantValue>>(value);
  if (!constant)
    return solver->propagateIfChanged(
        cv, cv->join(ConstantValue::getUnknownConstant()));

  // Dialect *dialect;
  // if (auto *parent = value.getDefiningOp())
  //   dialect = parent->getDialect();
  // else
  //   dialect = value.getParentBlock()->getParentOp()->getDialect();
  // solver->propagateIfChanged(
  //     cv, cv->join(ConstantValue(FloatAttr::get(value.getType(), *constant),
  //                                dialect)));
}

LogicalResult TaffoAffineRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const TaffoAffineRangeLattice *> operands,
    ArrayRef<TaffoAffineRangeLattice *> results) {
  // If the lattice on any operand is unitialized, bail out.
  if (llvm::any_of(operands, [](const TaffoAffineRangeLattice *lattice) {
        return lattice->getValue().isUninitialized();
      })) {
    return mlir::failure();
  }

  if (dyn_cast<LoopLikeOpInterface>(op)) {
    auto search = loops.find(op);
    // new loop found
    if (search == loops.end()) {
      loops.insert(std::make_pair(op, estimateTripCount(op)));
      LLVM_DEBUG(llvm::dbgs()
                 << "found new loop " << *op
                 << "\nwith trip count: " << estimateTripCount(op) << "\n");
    }
  }

  auto inferrable = dyn_cast<InferTaffoRangeInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return mlir::failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "[Affine VRA] Inferring aranges for " << *op
                          << "\n");
  SmallVector<Var> argRanges(
      llvm::map_range(operands, [](const TaffoAffineRangeLattice *val) {
        return val->getValue().getValue();
      }));

  auto parent = loops.find(op->getParentOp());
  auto search = opVisits.find(op);
  if (parent != loops.end()) {
    if (search != opVisits.end()) {
      search->second++;
    } else {
      opVisits.insert(std::make_pair(op, 0));
    }
  }

  if (search != opVisits.end() &&
      llvm::all_of(op->getOperands(),
                   [&](Value v) { return hitTripCount(v); }) &&
      llvm::all_of(op->getResults(),
                   [&](Value v) { return hitTripCount(v); })) {
    LLVM_DEBUG(llvm::dbgs() << "trip count hit for op " << *op << "\n");
    return mlir::failure();
  }

  auto joinCallback = [&](Value v, const Var &attrs) {
    // TODO handle function arguments
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(llvm::dbgs()
               << "[Affine VRA] Inferred range: ["
               << attrs.get_range().start.convertToDouble() << ", "
               << attrs.get_range().end.convertToDouble() << "]\n");
    LLVM_DEBUG(llvm::dbgs() << attrs.print_affine_form());
    TaffoAffineRangeLattice *lattice = results[result.getResultNumber()];
    TaffoAffineValueRange oldRange = lattice->getValue();

    ChangeResult changed = lattice->join(TaffoAffineValueRange{attrs});

    /*
     * for now we ignore loop variant ranges, as those will depend
     * on trip count estimation to be performed before VRA
     *
    // TODO: propagate trip count estimation to VRA
    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRange.isUninitialized() &&
        !(lattice->getValue() == oldRange)) {
      //LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
      changed |= lattice->join(TaffoAffineValueRange::getMaxRange(v));
    }
    */
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferTaffoAffineRanges(argRanges, joinCallback);
  return mlir::success();
}

void TaffoAffineRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<TaffoAffineRangeLattice *> argLattices, unsigned firstIndex) {

  if (dyn_cast<LoopLikeOpInterface>(op)) {
    auto search = loops.find(op);
    // new loop found
    if (search == loops.end() && op->getNumResults() > 0 &&
        llvm::isa<taffo::RealType>(op->getResultTypes().front())) {
      loops.insert(std::make_pair(op, estimateTripCount(op)));
      LLVM_DEBUG(llvm::dbgs()
                 << "found new loop " << *op
                 << "\nwith trip count: " << estimateTripCount(op) << "\n");
    }
  }

  if (auto inferrable = dyn_cast<InferTaffoRangeInterface>(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[Affine VRA] Inferring ranges for " << *op << "\n");
    // If the lattice on any operand is unitialized, bail out.
    if (llvm::any_of(op->getOperands(), [&](Value value) {
          return getLatticeElementFor(op, value)->getValue().isUninitialized();
        }))
      return;
    SmallVector<Var> argRanges(
        llvm::map_range(op->getOperands(), [&](Value value) {
          return getLatticeElementFor(op, value)->getValue().getValue();
        }));

    auto parent = loops.find(op->getParentOp());
    auto search = opVisits.find(op);
    if (parent != loops.end()) {
      if (search != opVisits.end()) {
        search->second++;
      } else {
        opVisits.insert(std::make_pair(op, 0));
      }
    }

    if (search != opVisits.end() &&
        llvm::all_of(op->getOperands(),
                     [&](Value v) { return hitTripCount(v); }) &&
        llvm::all_of(op->getResults(),
                     [&](Value v) { return hitTripCount(v); })) {
      LLVM_DEBUG(llvm::dbgs() << "trip count hit for op " << *op << "\n");
      return;
    }

    auto joinCallback = [&](Value v, const Var &attrs) {
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg)
        return;
      if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
        return;

      LLVM_DEBUG(llvm::dbgs()
                 << "[Affine VRA] Inferred range: ["
                 << attrs.get_range().start.convertToDouble() << ", "
                 << attrs.get_range().end.convertToDouble() << "]\n");
      LLVM_DEBUG(llvm::dbgs() << attrs.print_affine_form());
      TaffoAffineRangeLattice *lattice = argLattices[arg.getArgNumber()];
      TaffoAffineValueRange oldRange = lattice->getValue();

      ChangeResult changed = lattice->join(TaffoAffineValueRange{attrs});

      /*
       * for now we ignore loop variant ranges, as those will depend
       * on trip count estimation to be performed before VRA
       *
      // TODO: propagate trip count estimation to VRA
      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).
      bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<OpTrait::IsTerminator>();
      });
      if (isYieldedValue && !oldRange.isUninitialized() &&
          !(lattice->getValue() == oldRange)) {
        //LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
        changed |= lattice->join(TaffoAffineValueRange::getMaxRange(v));
      }

       */
      propagateIfChanged(lattice, changed);
    };

    inferrable.inferTaffoAffineRanges(argRanges, joinCallback);
    return;
  }

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}
