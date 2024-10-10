// most includes copied from mlir/Analysis/DataFlow/IntegerRangeAnalysis.cpp
// might want to clean up later

#include "Taffo/Transforms/NtvRangeAnalysis.h"
#include "Taffo/Dialect/Ops.h"
#include "Taffo/Dialect/Taffo.h"
#include "Taffo/Interfaces/InferTaffoRangeNtvInterface.h"

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
#include "llvm/Support/Debug.h"
#include <cassert>
#include <llvm/ADT/APFloat.h>
#include <optional>
#include <utility>

#define DEBUG_TYPE "value-range-analysis"

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::taffo;

TaffoValueRange TaffoValueRange::getMaxRange(Value value) {

  // placeholder, needs to work based on annotations of Value in the future
  // instead of defaulting to APFloat::IEEEdouble()
  // should this return [-max, max] instead?
  return TaffoValueRange(
      NtvRange(APFloat::getInf(APFloat::IEEEdouble(), true),
               APFloat::getInf(APFloat::IEEEdouble(), false)));
}

static int64_t estimateTripCount(Operation *op) {
  const int64_t defaultTripCount = 1000;

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

bool TaffoNtvRangeAnalysis::hitTripCount(Value v) {
  LLVM_DEBUG(llvm::dbgs() << "checking trip count for " << v << "\n\n");

  auto arg = dyn_cast<BlockArgument>(v);
  if (arg)
    return true;

  auto defOp = v.getDefiningOp();
  if (!defOp) {
    LLVM_DEBUG(llvm::dbgs() << "no defining op \n");
    return true;
  }
  LLVM_DEBUG(llvm::dbgs() << "defining op: " << *defOp << "\n");

  auto parent = defOp->getParentOp();
  if (!parent) {
    LLVM_DEBUG(llvm::dbgs() << "no parent op \n");
    return true;
  }
  LLVM_DEBUG(llvm::dbgs() << "parent op: " << *parent << "\n");

  auto searchParent = this->loops.find(parent);
  // if parent is not in loop map there's no need to track visits
  if (searchParent == this->loops.end()) {
    LLVM_DEBUG(llvm::dbgs() << "parent is not loop\n");
    return true;
  }

  auto searchOp = this->opVisits.find(defOp);
  // if the op is not in opVisits, it doesn't have a trip count
  if (searchOp == this->opVisits.end()) {
    LLVM_DEBUG(llvm::dbgs() << "op not in op visits\n");
    return true;
  }
  LLVM_DEBUG(llvm::dbgs() << "op trip count: " << searchOp->second << "\n");
  LLVM_DEBUG(llvm::dbgs() << "parent trip count: " << searchParent->second
                          << "\n");
  return searchOp->second >= searchParent->second;
}

void TaffoRangeLattice::onUpdate(DataFlowSolver *solver) const {
  Lattice::onUpdate(solver);

  // If the range can be narrowed to a constant, update the constant
  // value of the SSA value.
  const NtvRange range = getValue().getValue();
  std::optional<APFloat> constant = range.first == range.second
                                        ? std::optional<APFloat>(range.first)
                                        : std::nullopt;

  auto value = point.get<Value>();
  auto *cv = solver->getOrCreateState<Lattice<ConstantValue>>(value);
  if (!constant)
    return solver->propagateIfChanged(
        cv, cv->join(ConstantValue::getUnknownConstant()));

  Dialect *dialect;
  if (auto *parent = value.getDefiningOp())
    dialect = parent->getDialect();
  else
    dialect = value.getParentBlock()->getParentOp()->getDialect();
  solver->propagateIfChanged(
      cv, cv->join(ConstantValue(FloatAttr::get(value.getType(), *constant),
                                 dialect)));
}

void TaffoNtvRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const TaffoRangeLattice *> operands,
    ArrayRef<TaffoRangeLattice *> results) {
  // If the lattice on any operand is unitialized, bail out.
  if (llvm::any_of(operands, [](const TaffoRangeLattice *lattice) {
        return lattice->getValue().isUninitialized();
      })) {
    LLVM_DEBUG(llvm::dbgs() << "uninitialized operands in op " << *op << "\n");
    return;
  }

  if (dyn_cast<LoopLikeOpInterface>(op)) {
    auto search = loops.find(op);
    // new loop found
    if (search == loops.end() &&
        llvm::isa<taffo::RealType>(op->getResultTypes().front())) {
      loops.insert(std::make_pair(op, estimateTripCount(op)));
      LLVM_DEBUG(llvm::dbgs()
                 << "found new loop " << *op
                 << "\nwith trip count: " << estimateTripCount(op) << "\n");
    }
  }

  auto inferrable = dyn_cast<InferTaffoRangeNtvInterface>(op);
  if (!inferrable)
    return setAllToEntryStates(results);

  LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
  SmallVector<NtvRange> argRanges(
      llvm::map_range(operands, [](const TaffoRangeLattice *val) {
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
    return;
  }

  auto joinCallback = [&](Value v, const NtvRange &attrs) {
    // TODO handle function arguments
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(llvm::dbgs()
               << "Inferred range: [" << attrs.first.convertToDouble() << ", "
               << attrs.second.convertToDouble() << "]\n");
    TaffoRangeLattice *lattice = results[result.getResultNumber()];
    TaffoValueRange oldRange = lattice->getValue();

    ChangeResult changed = lattice->join(TaffoValueRange{attrs});

    // TODO: propagate trip count estimation to VRA
    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).

    // bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
    //   return op->hasTrait<OpTrait::IsTerminator>();
    // });
    // if (isYieldedResult && !oldRange.isUninitialized() &&
    //     !(lattice->getValue() == oldRange)) {
    //   LLVM_DEBUG(llvm::dbgs()
    //              << "Loop variant loop result detected in op " << *op <<
    //              "\n");
    //   // changed |= lattice->join(TaffoValueRange::getMaxRange(v));
    //   //  ignore loops
    //   Value *p_v = &v;
    //   auto search = loopVisits.find(p_v);
    //   if (search != loopVisits.end()) {
    //     changed = mlir::ChangeResult::NoChange;
    //   } else {
    //     changed = mlir::ChangeResult::Change;
    //     loopVisits.insert(std::pair(p_v, 0));
    //   }
    // }

    // check trip count for ops in loop body
    // if (auto parent = dyn_cast<LoopLikeOpInterface>(op->getParentOp())) {
    //  LLVM_DEBUG(llvm::dbgs()
    //                 << "found loop body op " << *op->getParentOp() << "\nwith
    //                 "
    //                 << loopVisits.find(parent)->second << " remaining iters"
    //                 << "\n");
    //  if (loopVisits.find(parent)->second == 0) {
    //    changed = mlir::ChangeResult::NoChange;
    //  }
    //}
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferTaffoRanges(argRanges, joinCallback);
}

void TaffoNtvRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<TaffoRangeLattice *> argLattices, unsigned firstIndex) {
  LLVM_DEBUG(llvm::dbgs() << "visitNonControlFlowArguments call " << *op
                          << "\n");

  if (dyn_cast<LoopLikeOpInterface>(op)) {
    auto search = loops.find(op);
    // new loop found
    LLVM_DEBUG(llvm::dbgs() << "searching for loop of type"
                            << op->getResultTypes().front() << "\n");
    if (search == loops.end() &&
        llvm::isa<taffo::RealType>(op->getResultTypes().front())) {
      loops.insert(std::make_pair(op, estimateTripCount(op)));
      LLVM_DEBUG(llvm::dbgs()
                 << "found new loop " << *op
                 << "\nwith trip count: " << estimateTripCount(op) << "\n");
    }
  }

  if (auto inferrable = dyn_cast<InferTaffoRangeNtvInterface>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
    // If the lattice on any operand is unitialized, bail out.
    if (llvm::any_of(op->getOperands(), [&](Value value) {
          return getLatticeElementFor(op, value)->getValue().isUninitialized();
        }))
      return;
    SmallVector<NtvRange> argRanges(
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

    auto joinCallback = [&](Value v, const NtvRange &attrs) {
      // TODO handle function arguments
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg)
        return;
      if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
        return;

      LLVM_DEBUG(llvm::dbgs()
                 << "Inferred range: [" << attrs.first.convertToDouble() << ", "
                 << attrs.second.convertToDouble() << "]\n");
      TaffoRangeLattice *lattice = argLattices[arg.getArgNumber()];
      TaffoValueRange oldRange = lattice->getValue();

      ChangeResult changed = lattice->join(TaffoValueRange{attrs});

      // TODO: propagate trip count estimation to VRA
      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).

      // bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      //   return op->hasTrait<OpTrait::IsTerminator>();
      // });
      // if (isYieldedResult && !oldRange.isUninitialized() &&
      //     !(lattice->getValue() == oldRange)) {
      //   LLVM_DEBUG(llvm::dbgs()
      //              << "Loop variant loop result detected in op " << *op <<
      //              "\n");
      //   // changed |= lattice->join(TaffoValueRange::getMaxRange(v));
      //   //  ignore loops
      //   Value *p_v = &v;
      //   auto search = loopVisits.find(p_v);
      //   if (search != loopVisits.end()) {
      //     changed = mlir::ChangeResult::NoChange;
      //   } else {
      //     changed = mlir::ChangeResult::Change;
      //     loopVisits.insert(std::pair(p_v, 0));
      //   }
      // }
      propagateIfChanged(lattice, changed);
    };

    inferrable.inferTaffoRanges(argRanges, joinCallback);
    return;
  }

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}
