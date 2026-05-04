#include "Taffo/Transforms/RaiseToTaffoPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeRange.h"
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

  class ArithToTaffoTypeConverter : public mlir::TypeConverter {
  public:
    ArithToTaffoTypeConverter(MLIRContext *ctx) {
      addConversion([](Type type) { return type; });
      addConversion([ctx](FloatType type) -> Type {
        if (type.isF32())
          return taffo::RealType::get(ctx, false, 0, 0);
        return type;
      });

      addTargetMaterialization(
          [&](mlir::OpBuilder &builder, mlir::TypeRange resultType,
              mlir::ValueRange inputs,
              mlir::Location loc) -> llvm::SmallVector<mlir::Value> {
            if (inputs.size() != 1) {
              return {};
            }

            auto castToRealOp =
                builder.create<mlir::UnrealizedConversionCastOp>(
                    loc, resultType, inputs);

            llvm::SmallVector<mlir::Value> result;
            result.push_back(castToRealOp.getResult(0));
            return result;
          });

      addSourceMaterialization(
          [&](mlir::OpBuilder &builder, mlir::Type resultType,
              mlir::ValueRange inputs,
              mlir::Location loc) -> mlir::Value {
            if (inputs.size() != 1) {
              return {};
            }
            auto castToRealOp =
                builder.create<mlir::UnrealizedConversionCastOp>(
                    loc, resultType, inputs);

            return castToRealOp.getResult(0);
          });
    }
  };
  struct RewriteFor : public OpConversionPattern<scf::ForOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      Block *body = op.getBody();

      Region &region = op->getRegion(0);

      rewriter.startOpModification(op);
      auto terminator = cast<scf::YieldOp>(body->getTerminator());
      SmallVector<Value> terminatorRes;
      if (failed(rewriter.getRemappedValues(terminator->getOperands(),
                                            terminatorRes)))
        return failure();
      rewriter.modifyOpInPlace(terminator,
                               [&] { terminator->setOperands(terminatorRes); });

      rewriter.finalizeOpModification(op);

      if (failed(rewriter.convertRegionTypes(&region, *getTypeConverter())))
        return failure();

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      auto newOp =
          b.create<scf::ForOp>(adaptor.getLowerBound(), adaptor.getUpperBound(),
                               adaptor.getStep(), adaptor.getInitArgs());

      // We do not need the empty block created by rewriter.
      rewriter.eraseBlock(newOp.getBody(0));
      // Inline the type converted region from the original operation.
      rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                  newOp.getRegion().end());

      rewriter.replaceOp(op, newOp);
      return success();
    }
  };

  struct RewriteSetRangeCall : public OpConversionPattern<func::CallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

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

      rewriter.replaceOp(op, cast2real.getResult());
      return success();
    }
  };

  struct RewriteArithAddOp : public OpConversionPattern<arith::AddFOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      OpBuilder builder(op);

      auto loc = op.getLoc();
      auto input1 = adaptor.getLhs();
      auto input2 = adaptor.getRhs();

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

  struct RewriteArithSubOp : public OpConversionPattern<arith::SubFOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(arith::SubFOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      OpBuilder builder(op);

      auto loc = op.getLoc();
      auto input1 = adaptor.getLhs();
      auto input2 = adaptor.getRhs();

      auto return_type =
          taffo::RealType::get(builder.getContext(), false, 0, 0);
      // Create a taffo.sub operation
      auto subOp =
          rewriter.create<taffo::SubOp>(loc, return_type, input1, input2);

      LLVM_DEBUG(llvm::dbgs() << "Created sub:" << subOp << "\n");

      // Replace the original operation with the new one
      rewriter.replaceOp(op, subOp.getResult());
      return success();
    }
  };

  struct RewriteArithMulOp : public OpConversionPattern<arith::MulFOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(arith::MulFOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      OpBuilder builder(op);

      auto loc = op.getLoc();
      auto input1 = adaptor.getLhs();
      auto input2 = adaptor.getRhs();

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

  struct RewriteArithDivOp : public OpConversionPattern<arith::DivFOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(arith::DivFOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      auto loc = op.getLoc();
      auto input1 = adaptor.getLhs();
      auto input2 = adaptor.getRhs();

      auto returnType =
          taffo::RealType::get(rewriter.getContext(), false, 0, 0);
      // Create a taffo.div operation.
      auto divOp =
          rewriter.create<taffo::DivOp>(loc, returnType, input1, input2);

      rewriter.replaceOp(op, divOp.getResult());
      return success();
    }
  };

  struct InsertCast2FloatReturnOp : public OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      OpBuilder builder(op);
      SmallVector<Value> newOperands;
      bool changed = false;
      // create new func.return

      // Iterate over each operand in tandem with the declared result type.
      for (auto it : llvm::enumerate(op.getOperands())) {
        unsigned i = it.index();
        Value operand = it.value();

        // if operand is of type f32 insert an unrealized conversion to
        // taffo.real
        auto opType = operand.getType();
        if (opType.isF32()) {
          // Print the operands defining op
          LLVM_DEBUG(llvm::dbgs() << "Operand: " << operand << "\n");
          LLVM_DEBUG(llvm::dbgs() << "Operand defining op: "
                                  << operand.getDefiningOp() << "\n");
          auto realTy =
              taffo::RealType::get(rewriter.getContext(), false, 0, 0);
          auto castToReal = builder.create<mlir::UnrealizedConversionCastOp>(
              op.getLoc(), realTy, operand);
          operand = castToReal.getResult(0);

          auto cast2float = builder.create<taffo::CastToFloatOp>(
              op.getLoc(), opType, operand);
          newOperands.push_back(cast2float.getResult());
          changed = true;
        }

        else {
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
      auto newOp = builder.create<func::ReturnOp>(op.getLoc(), newOperands);
      rewriter.replaceOp(op, newOp);

      return success();
    }
  };

  struct InsertCast2FloatFuncCallOp : public OpConversionPattern<func::CallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
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

  // struct RewriteYieldOp : public OpConversionPattern<scf::YieldOp> {
  //   using OpConversionPattern::OpConversionPattern;
  //   LogicalResult
  //   matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
  //                   ConversionPatternRewriter &rewriter) const override {
  //     // If there are no f32 operands, let other patterns handle it.
  //     bool hasF32 = llvm::any_of(op.getOperands(),
  //                                [&](Value v) { return v.getType().isF32();
  //                                });
  //     if (!hasF32)
  //       return failure();

  //     // The parent block of this yield has already had its signature
  //     // rewritten, so its block args are now !taffo.real.
  //     Block *parent = op->getBlock();
  //     unsigned numArgs = parent->getNumArguments();

  //     // Build the new operand list: for each original operand:
  //     //   - if it was a block argument, grab the *converted* block arg
  //     //   - otherwise just reuse the same SSA value (it won’t be f32)
  //     SmallVector<Value, 4> newOperands;
  //     newOperands.reserve(op.getNumOperands());
  //     for (auto opOperand : llvm::enumerate(op.getOperands())) {
  //       Value oldVal = opOperand.value();
  //       if (BlockArgument *arg = &oldVal.dyn_cast<BlockArgument>()) {
  //         // map the old arg # to the *same* new block argument
  //         newOperands.push_back(parent->getArgument(arg->getArgNumber()));
  //       } else {
  //         newOperands.push_back(oldVal);
  //       }
  //     }

  //     // Replace with a fresh scf.yield that returns those block args
  //     rewriter.replaceOpWithNewOp<scf::YieldOp>(op, newOperands);
  //     return success();
  //   }
  // };

  struct RewriteIfOp : public OpConversionPattern<scf::IfOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // Convert the types of the then region.
      Region &thenRegion = op.getThenRegion();
      if (failed(rewriter.convertRegionTypes(&thenRegion, *getTypeConverter())))
        return failure();

      // Determine if an else region exists.
      bool hasElse = !op.getElseRegion().empty();

      // Update then region terminator operands.
      Block &thenBlock = thenRegion.getBlocks().back();
      if (auto thenYield =
              llvm::dyn_cast<scf::YieldOp>(thenBlock.getTerminator())) {
        SmallVector<Value> newOperands;
        if (failed(rewriter.getRemappedValues(thenYield->getOperands(),
                                              newOperands)))
          return failure();
        rewriter.modifyOpInPlace(
            thenYield, [&]() { thenYield->setOperands(newOperands); });
      }

      // If an else region exists, update its terminator operands.
      if (hasElse) {
        Region &elseRegion = op.getElseRegion();
        Block &elseBlock = elseRegion.getBlocks().back();
        if (auto elseYield =
                llvm::dyn_cast<scf::YieldOp>(elseBlock.getTerminator())) {
          SmallVector<Value> newOperands;
          if (failed(rewriter.getRemappedValues(elseYield->getOperands(),
                                                newOperands)))
            return failure();
          rewriter.modifyOpInPlace(
              elseYield, [&]() { elseYield->setOperands(newOperands); });
        }
      }

      // If an else region exists, convert its types as well.
      if (hasElse) {
        Region &elseRegion = op.getElseRegion();
        if (failed(
                rewriter.convertRegionTypes(&elseRegion, *getTypeConverter())))
          return failure();
      }

      // Create a new scf.if op with the same condition and result types.
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      auto resType = taffo::RealType::get(b.getContext(), false, 0, 0);
      auto newOp = b.create<scf::IfOp>(op.getLoc(), resType,
                                       adaptor.getCondition(), hasElse);

      // Remove the automatically created then block and inline the converted
      // then region.
      rewriter.eraseBlock(newOp.thenBlock());
      rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                  newOp.getThenRegion().end());

      // If there is an else region, do the same for it.
      if (hasElse) {
        rewriter.eraseBlock(newOp.elseBlock());
        rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                    newOp.getElseRegion().end());
      }
      rewriter.replaceOp(op, newOp.getResults());
      return success();
    }
  };

  struct RemoveSetRangeDef : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      // rewriter.startOpModification(op);
      rewriter.eraseOp(op);
      return success();
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    ArithToTaffoTypeConverter typeConverter(context);

    // dialect conversion driver
    // mark all operations that have f32 args as ilegal
    target.addLegalDialect<TaffoDialect>();
    target.addLegalOp<scf::YieldOp>();
    target.addDynamicallyLegalOp<arith::AddFOp, arith::SubFOp, arith::MulFOp,
                                 arith::DivFOp, func::ReturnOp>(
        [](Operation *op) {
          // Check if the operation has any f32 arguments
          for (auto operand : op->getOperands()) {
            if (operand.getType().isF32()) {
              LLVM_DEBUG(llvm::dbgs() << "Found f32 operand in op: " << *op);
              return false; // Mark as illegal
            }
          }
          return true; // Mark as legal
        });

    target.addDynamicallyLegalOp<func::CallOp>([](Operation *op) {
      // cast to CallOp
      auto callOp = llvm::dyn_cast<func::CallOp>(op);
      // Check if the function being called is a taffo function
      return !callOp.getCallee().contains("set_range");
    });

    target.addDynamicallyLegalOp<func::FuncOp>([](Operation *op) {
      // cast to CallOp
      auto funcOp = llvm::dyn_cast<func::FuncOp>(op);
      // Check if the function being called is a taffo function
      return !funcOp.getName().contains("set_range");
    });

    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) { return typeConverter.isLegal(op); });

    target.addDynamicallyLegalOp<scf::IfOp>(
        [&](scf::IfOp op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(&getContext());
    patterns.add<RewriteFor, RewriteSetRangeCall, RewriteArithAddOp, RewriteArithSubOp,
                 RewriteArithMulOp, RewriteArithDivOp, InsertCast2FloatReturnOp, RemoveSetRangeDef,RewriteIfOp /*,
                 InsertCast2FloatFuncCallOp, RewriteIfOp*/>(typeConverter,
                                                          context);
    // patterns.add<RewriteYieldOp>(patterns.getContext());
    (void)applyPartialConversion(getOperation(), target, std::move(patterns));

    // getOperation()->dump();
    // target.addDynamicallyLegalOp<scf::ForOp>(
    //     [&](scf::ForOp op) { return typeConverter.isLegal(op); });

    // RewritePatternSet loopPatterns(context);
    // loopPatterns.add<RewriteFor>(typeConverter, context);
    // if (failed(applyPartialConversion(getOperation(), target,
    //                                   std::move(loopPatterns)))) {
    //   signalPassFailure();
    // }
  }
};
} // namespace mlir
