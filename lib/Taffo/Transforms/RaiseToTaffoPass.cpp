#include "Taffo/Transforms/RaiseToTaffoPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Taffo/Dialect/Ops.h"

namespace mlir::taffo {
#define GEN_PASS_DEF_RAISETOTAFFOPASS
#include "Taffo/Transforms/Passes.h.inc"
} // namespace mlir::taffo

using namespace ::mlir::taffo;

namespace mlir {
class RaiseToTaffoPass
    : public mlir::taffo::impl::RaiseToTaffoPassBase<RaiseToTaffoPass> {
public:
  using RaiseToTaffoPassBase::RaiseToTaffoPassBase;
  struct RewriteSetRangeCall : public OpRewritePattern<func::CallOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {

      if (op.getCallee() == "set_range") {
        // Create a taffo.cast2real operation
        OpBuilder builder(op);
        auto loc = op.getLoc();
        auto input = op.getOperand(0);

        LLVM_DEBUG(llvm::dbgs() << "Input for set_range: " << input << "\n");

        // Get the second and third operands
        Value secondOperand = op.getOperand(1);
        Value thirdOperand = op.getOperand(2);
        Value fourthOperand = op.getOperand(3);

        // Check if each is defined by a ConstantOp
        auto secondConstOp = secondOperand.getDefiningOp<arith::ConstantOp>();
        auto thirdConstOp = thirdOperand.getDefiningOp<arith::ConstantOp>();
        auto fourthConstOp = fourthOperand.getDefiningOp<arith::ConstantOp>();

        if (!secondConstOp || !thirdConstOp || !fourthConstOp)
          return failure();

        // Extract min, max, and precision
        auto min = secondConstOp.getValue().dyn_cast<FloatAttr>();
        auto max = thirdConstOp.getValue().dyn_cast<FloatAttr>();
        auto precision = fourthConstOp.getValue().dyn_cast<FloatAttr>();
        if (!min || !max || !precision)
          return failure();

        auto return_type =
            taffo::RealType::get(builder.getContext(), false, 0, 0);
        LLVM_DEBUG(llvm::dbgs() << "Created type: " << return_type << "\n");
        LLVM_DEBUG(llvm::dbgs() << "Created min: " << min << "\n");
        LLVM_DEBUG(llvm::dbgs() << "Created max: " << max << "\n");
        LLVM_DEBUG(llvm::dbgs() << "Created precision: " << precision << "\n");

        auto cast2real = rewriter.create<taffo::CastToRealOp>(
            loc, return_type, input, precision, min, max);
        LLVM_DEBUG(llvm::dbgs() << "Created cast2real:" << cast2real << "\n");

        input.replaceUsesWithIf(cast2real, [&](OpOperand &operand) {
          return operand.getOwner() != cast2real.getOperation();
        });
        rewriter.eraseOp(op);
        return success();
      }
      return failure();
    }
  };

  struct RewriteArithAddOp : public OpRewritePattern<arith::AddFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddFOp op,
                                  PatternRewriter &rewriter) const override {

      OpBuilder builder(op);

      auto loc = op.getLoc();
      auto input1 = op.getOperand(0);
      auto input2 = op.getOperand(1);

      // Only perform the rewrite if all the args of the op are of type
      // taffo::RealType
      if (!input1.getType().isa<taffo::RealType>() ||
          !input2.getType().isa<taffo::RealType>()) {
        return failure();
      }

      auto return_type =
          taffo::RealType::get(builder.getContext(), false, 0, 0);
      // Create a taffo.add operation
      auto addOp =
          rewriter.create<taffo::AddOp>(loc, return_type, input1, input2);

      LLVM_DEBUG(llvm::dbgs() << "Created add:" << addOp << "\n");

      // Replace the original operation with the new one
      rewriter.replaceOp(op, addOp.getResult());
      return success();
    }
  };

  struct RewriteArithMulOp : public OpRewritePattern<arith::MulFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::MulFOp op,
                                  PatternRewriter &rewriter) const override {
      OpBuilder builder(op);

      auto loc = op.getLoc();
      auto input1 = op.getOperand(0);
      auto input2 = op.getOperand(1);

      // Only perform the rewrite if all the args of the op are of type
      // taffo::RealType
      if (!input1.getType().isa<taffo::RealType>() ||
          !input2.getType().isa<taffo::RealType>()) {
        return failure();
      }

      auto return_type =
          taffo::RealType::get(builder.getContext(), false, 0, 0);
      // Create a taffo.add operation
      auto mulOp =
          rewriter.create<taffo::MultOp>(loc, return_type, input1, input2);

      LLVM_DEBUG(llvm::dbgs() << "Created Mul:" << mulOp << "\n");

      // Replace the original operation with the new one
      rewriter.replaceOp(op, mulOp.getResult());
      return success();
    }
  };

  struct InsertCast2FloatReturnOp : public OpRewritePattern<func::ReturnOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(func::ReturnOp op,
                                  PatternRewriter &rewriter) const override {
      // Find the parent function and get its declared result types
      auto function = op->getParentOfType<func::FuncOp>();
      if (!function)
        return failure();

      auto funcType = function.getFunctionType();
      auto resultTypes = funcType.getResults();
      // If the number of return operands doesn't match the function signature,
      // bail out (or handle differently if your IR can mismatch).
      if (op.getNumOperands() != resultTypes.size())
        return failure();

      // We'll rebuild the operand list with cast ops where needed.
      SmallVector<Value> newOperands;
      newOperands.reserve(op.getNumOperands());

      bool changed = false;
      // Iterate over each operand in tandem with the declared result type.
      for (auto it : llvm::enumerate(op.getOperands())) {
        unsigned i = it.index();
        Value operand = it.value();
        Type desiredType = resultTypes[i];

        // If the function says this return value should be float,
        // and the operand is not *already* the correct cast,
        // insert a CastToFloatOp with the declaredType.
        auto floatTy = desiredType.dyn_cast<FloatType>();
        if (floatTy && operand.getType().isa<taffo::RealType>()) {
          // Skip if operand is already from taffo.cast2float with matching
          // type.
          if (auto castOp = operand.getDefiningOp<taffo::CastToFloatOp>()) {
            if (castOp.getResult().getType() == floatTy) {
              newOperands.push_back(operand);
              continue;
            }
          }
          // Otherwise, create a new cast
          auto loc = op.getLoc();
          auto cast2float =
              rewriter.create<taffo::CastToFloatOp>(loc, floatTy, operand);
          newOperands.push_back(cast2float.getResult());
          changed = true;
        } else {
          // If it's not a float return type (e.g. i32), we won't cast.
          newOperands.push_back(operand);
        }
      }

      // If we didn't actually change anything, tell the rewriter the pattern
      // failed so it can move on to other patterns or avoid re-matching
      // forever.
      if (!changed)
        return failure();

      // Update the return op in-place with the new operands.
      rewriter.startOpModification(op);
      op->setOperands(newOperands);
      rewriter.finalizeOpModification(op);

      return success();
    }
  };

  struct InsertCast2FloatFuncCallOp : public OpRewritePattern<func::CallOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {
      // Get the function being called
      auto function = op.getCallee();
      LLVM_DEBUG(llvm::dbgs() << "Function: " << function << "\n");
      auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          op, StringAttr::get(op.getContext(), function));
      if (!funcOp)
        return failure();

      auto funcType = funcOp.getFunctionType();
      auto inputTypes = funcType.getInputs();

      // If the number of call operands doesn't match the function signature,
      // bail out (or handle differently if your IR can mismatch).
      if (op.getNumOperands() != inputTypes.size())
        return failure();

      // We'll rebuild the operand list with cast ops where needed.
      SmallVector<Value> newOperands;
      newOperands.reserve(op.getNumOperands());

      bool changed = false;
      // Iterate over each operand in tandem with the declared input type.
      for (auto it : llvm::enumerate(op.getOperands())) {
        unsigned i = it.index();
        Value operand = it.value();
        Type desiredType = inputTypes[i];

        // If the function says this input value should be float,
        // and the operand is not *already* the correct cast,
        // insert a CastToFloatOp with the declaredType.
        auto floatTy = desiredType.dyn_cast<FloatType>();
        if (floatTy && operand.getType().isa<taffo::RealType>()) {
          // Skip if operand is already from taffo.cast2float with matching
          // type.
          if (auto castOp = operand.getDefiningOp<taffo::CastToFloatOp>()) {
            if (castOp.getResult().getType() == floatTy) {
              newOperands.push_back(operand);
              continue;
            }
          }
          // Otherwise, create a new cast
          auto loc = op.getLoc();
          auto cast2float =
              rewriter.create<taffo::CastToFloatOp>(loc, floatTy, operand);
          newOperands.push_back(cast2float.getResult());
          changed = true;
        } else {
          // If it's not a float input type (e.g. i32), we won't cast.
          newOperands.push_back(operand);
        }
      }

      // If we didn't actually change anything, tell the rewriter the pattern
      // failed so it can move on to other patterns or avoid re-matching
      // forever.
      if (!changed)
        return failure();

      // Update the call op in-place with the new operands.
      rewriter.startOpModification(op);
      op->setOperands(newOperands);
      rewriter.finalizeOpModification(op);

      return success();
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    // dialect conversion driver
    RewritePatternSet patterns(&getContext());
    patterns.add<RewriteSetRangeCall>(patterns.getContext());
    patterns.add<RewriteArithAddOp>(patterns.getContext());
    patterns.add<RewriteArithMulOp>(patterns.getContext());
    // patterns.add<InsertCast2FloatReturnOp>(patterns.getContext());
    // patterns.add<InsertCast2FloatFuncCallOp>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace mlir