// includes copied from mlir/Analysis/DataFlow/IntegerRangeAnalysis.cpp
// might want to clean up later

#include "Taffo/NtvRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <optional>
#include <utility>
#include <llvm/ADT/APFloat.h>

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::taffo;

TaffoValueRange TaffoValueRange::getMaxRange(Value value) {

  // placeholder, needs to work based on annotations of Value in the future
  // instead of defaulting to APFloat::IEEEdouble()
  // should this return [-max, max] instead?
  return TaffoValueRange(NtvRange(APFloat::getInf(APFloat::IEEEdouble(), true),
                                  APFloat::getInf(APFloat::IEEEdouble(), false)));
}

void TaffoRangeLattice::onUpdate(DataFlowSolver *solver) const {
  Lattice::onUpdate(solver);

  // If the range can be narrowed to a constant, update the constant
  // value of the SSA value.
  const NtvRange range = getValue().getValue();
  std::optional<APFloat> constant =
      range.first == range.second ? std::optional<APFloat>(range.first)
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
    return;
  }
  
  auto inferrable = dyn_cast<InferTaffoRangeNtvInterface>(op);
  if (!inferrable)
    return setAllToEntryStates(results);

  //LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
  SmallVector<NtvRange> argRanges(
      llvm::map_range(operands, [](const TaffoRangeLattice *val) {
        return val->getValue().getValue();
      }));

  auto joinCallback = [&](Value v, const NtvRange &attrs) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    //LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
    TaffoRangeLattice *lattice = results[result.getResultNumber()];
    TaffoValueRange oldRange = lattice->getValue();

    ChangeResult changed = lattice->join(TaffoValueRange{attrs});

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
      changed |= lattice->join(TaffoValueRange::getMaxRange(v));
    }
    */
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferTaffoRanges(argRanges, joinCallback);
}

void TaffoNtvRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<TaffoRangeLattice *> argLattices, unsigned firstIndex) {
  if (auto inferrable = dyn_cast<InferTaffoRangeNtvInterface>(op)) {
    //LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
    // If the lattice on any operand is unitialized, bail out.
    if (llvm::any_of(op->getOperands(), [&](Value value) {
          return getLatticeElementFor(op, value)->getValue().isUninitialized();
        }))
      return;
    SmallVector<NtvRange> argRanges(
        llvm::map_range(op->getOperands(), [&](Value value) {
          return getLatticeElementFor(op, value)->getValue().getValue();
        }));

    auto joinCallback = [&](Value v, const NtvRange &attrs) {
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg)
        return;
      if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
        return;

      //LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
      TaffoRangeLattice *lattice = argLattices[arg.getArgNumber()];
      TaffoValueRange oldRange = lattice->getValue();

      ChangeResult changed = lattice->join(TaffoValueRange{attrs});

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
        changed |= lattice->join(TaffoValueRange::getMaxRange(v));
      }

       */
      propagateIfChanged(lattice, changed);
    };

    inferrable.inferTaffoRanges(argRanges, joinCallback);
    return;
  }

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}

