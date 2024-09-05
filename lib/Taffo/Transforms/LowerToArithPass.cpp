#include "Taffo/Transforms/LowerToArithPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Taffo/Dialect/Ops.h"

namespace mlir::taffo {
#define GEN_PASS_DEF_LOWERTOARITHPASS
#include "Taffo/Transforms/Passes.h.inc"
} // namespace mlir::taffo

using namespace ::mlir::taffo;

namespace mlir {
class LowerToArithPass
    : public mlir::taffo::impl::LowerToArithPassBase<LowerToArithPass> {
public:
  using LowerToArithPassBase::LowerToArithPassBase;

  class TaffoToArithTypeConverter : public mlir::TypeConverter {
  public:
    TaffoToArithTypeConverter(MLIRContext *ctx) {
      addConversion([](Type type) { return type; });
      addConversion([ctx](RealType type) -> Type {
        return IntegerType::get(ctx, type.getBitwidth(),
                                IntegerType::SignednessSemantics::Signless);
      });

      addTargetMaterialization(
          [&](mlir::OpBuilder &builder, mlir::Type resultType,
              mlir::ValueRange inputs,
              mlir::Location loc) -> std::optional<mlir::Value> {
            if (inputs.size() != 1) {
              return std::nullopt;
            }

            auto CastToRealOp =
                builder.create<mlir::UnrealizedConversionCastOp>(
                    loc, resultType, inputs);

            return CastToRealOp.getResult(0);
          });

      addSourceMaterialization(
          [&](mlir::OpBuilder &builder, mlir::Type resultType,
              mlir::ValueRange inputs,
              mlir::Location loc) -> std::optional<mlir::Value> {
            if (inputs.size() != 1) {
              return std::nullopt;
            }

            auto CastToRealOp =
                builder.create<mlir::UnrealizedConversionCastOp>(
                    loc, resultType, inputs);

            return CastToRealOp.getResult(0);
          });
    }
  };

  // TODO handle function arguments
  static int getExp(Value v) {
    return ::llvm::dyn_cast<RealType>(v.getType()).getExponent();
  }

  static bool getSignd(Value v) {
    return ::llvm::dyn_cast<RealType>(v.getType()).getSignd();
  }

  struct ConvertAdd : public OpConversionPattern<AddOp> {
    ConvertAdd(mlir::MLIRContext *context)
        : OpConversionPattern<AddOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(AddOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      const int targetWidth = 32;

      auto buildIntAttr = [targetWidth](Builder b,
                                        int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(targetWidth), value);
      };

      RealType resType =
          ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

      int rhsExp = getExp(op.getRhs());
      int lhsExp = getExp(op.getLhs());
      Value rhs = adaptor.getRhs();
      Value lhs = adaptor.getLhs();

      // reconcile arguments of different signedness
      if (resType.getSignd() &&
          (getSignd(op.getRhs()) != getSignd(op.getLhs()))) {

        if (!getSignd(op.getRhs())) {
          rhsExp += 1;
          rhs = b.create<arith::ShRSIOp>(
                     rhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                    .getResult();
        }

        if (!getSignd(op.getRhs())) {
          lhsExp += 1;
          lhs = b.create<arith::ShRSIOp>(
                     lhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                    .getResult();
        }
      }

      int expDiff = std::abs(rhsExp - lhsExp);
      // if the difference in exponent is large enough that largest number of
      // the smaller operand cannot be represented by the larger operand,
      // we delete the op
      if (expDiff > targetWidth) {
        Value maxExpArg = rhsExp > lhsExp ? rhs : lhs;
        rewriter.replaceOp(op, maxExpArg);
        return success();
      }

      Value res;

      if (expDiff != 0) {
        Value to_shift = rhsExp < lhsExp ? rhs : lhs;
        Value no_shift = rhsExp > lhsExp ? rhs : lhs;

        arith::ConstantOp shift_amount =
            b.create<arith::ConstantOp>(buildIntAttr(b, expDiff));
        Value ShOp =
            resType.getSignd()
                ? b.create<arith::ShRSIOp>(to_shift, shift_amount).getResult()
                : b.create<arith::ShRUIOp>(to_shift, shift_amount).getResult();
        res = b.create<arith::AddIOp>(no_shift, ShOp);
      } else {
        res = b.create<arith::AddIOp>(lhs, rhs);
      }

      int resExpDiff = resType.getExponent() - std::max(rhsExp, lhsExp);

      if (resExpDiff == 0) {
        rewriter.replaceOp(op, res);
        return success();
      }

      arith::ConstantOp align_res =
          b.create<arith::ConstantOp>(buildIntAttr(b, std::abs(resExpDiff)));
      res = resExpDiff > 0
                ? resType.getSignd()
                      ? b.create<arith::ShRSIOp>(res, align_res).getResult()
                      : b.create<arith::ShRUIOp>(res, align_res).getResult()
                : b.create<arith::ShLIOp>(res, align_res).getResult();
      rewriter.replaceOp(op, res);
      return success();
    }
  };

  struct ConvertMult : public OpConversionPattern<MultOp> {
    ConvertMult(mlir::MLIRContext *context)
        : OpConversionPattern<MultOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(MultOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      const int targetWidth = 32;

      auto buildIntAttr = [targetWidth](Builder b,
                                        int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(targetWidth), value);
      };

      RealType resType =
          ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

      int rhsExp = getExp(op.getLhs());
      int lhsExp = getExp(op.getLhs());
      Value rhs = adaptor.getRhs();
      Value lhs = adaptor.getLhs();

      // reconcile arguments of different signedness
      if (resType.getSignd() &&
          (getSignd(op.getRhs()) != getSignd(op.getLhs()))) {

        if (!getSignd(op.getRhs())) {
          rhsExp += 1;
          rhs = b.create<arith::ShRSIOp>(
                     rhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                    .getResult();
        }

        if (!getSignd(op.getRhs())) {
          lhsExp += 1;
          lhs = b.create<arith::ShRSIOp>(
                     lhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                    .getResult();
        }
      }

      int implicitExp = rhsExp + lhsExp + resType.getBitwidth();
      int expDiff = resType.getExponent() - implicitExp;

      Value res = resType.getSignd()
                      ? b.create<arith::MulSIExtendedOp>(lhs, rhs).getHigh()
                      : b.create<arith::MulUIExtendedOp>(lhs, rhs).getHigh();

      if (expDiff == 0) {
        rewriter.replaceOp(op, res);
        return success();
      }

      arith::ConstantOp align_res =
          b.create<arith::ConstantOp>(b.getIntegerAttr(
              b.getIntegerType(resType.getBitwidth()), std::abs(expDiff)));
      res = expDiff > 0
                ? resType.getSignd()
                      ? b.create<arith::ShRSIOp>(res, align_res).getResult()
                      : b.create<arith::ShRUIOp>(res, align_res).getResult()
                : b.create<arith::ShLIOp>(res, align_res).getResult();
      rewriter.replaceOp(op, res);
      return success();
    }
  };

  struct ConvertCastToReal : public OpConversionPattern<CastToRealOp> {
    ConvertCastToReal(mlir::MLIRContext *context)
        : OpConversionPattern<CastToRealOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CastToRealOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      FloatType fType = ::llvm::dyn_cast<FloatType>(op.getFrom().getType());
      int ogWidth = fType.getWidth();

      if (ogWidth > 64) {
        op->emitOpError()
            << "Conversion from floats bigger than f64 is not yet supported";
        return failure();
      }

      RealType resType =
          ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

      if (resType.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion to fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      auto buildIntAttr = [ogWidth](Builder b, int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(ogWidth), value);
      };

      // imagine a world where you are a programmer who decides the method
      // "getFPMantissaWidth" returns the width of the mantissa field + 1
      // because it accounts for the implicit bit for some godforsaken reason
      int mantissaBitwidth = fType.getFPMantissaWidth() - 1;

      IntegerType destType = b.getIntegerType(resType.getBitwidth());

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(ogWidth), op.getFrom());

      int normalized_exp = resType.getExponent();

      // compute actual exponent in place
      IntegerAttr dtExp = buildIntAttr(
          b, (uint64_t)(std::abs(normalized_exp) << mantissaBitwidth));
      arith::ConstantOp dtExp_const = b.create<arith::ConstantOp>(dtExp);
      Value actual_exp =
          normalized_exp > 0
              ? b.create<arith::SubIOp>(bitcast, dtExp_const).getResult()
              : b.create<arith::AddIOp>(bitcast, dtExp_const).getResult();

      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, actual_exp);

      Value res =
          (resType.getSignd())
              ? b.create<arith::FPToSIOp>(destType, bitcast_back).getResult()
              : b.create<arith::FPToUIOp>(destType, bitcast_back).getResult();

      rewriter.replaceOp(op, res);
      return success();
    }
  };

  struct ConvertCastToFloat : public OpConversionPattern<CastToFloatOp> {
    ConvertCastToFloat(mlir::MLIRContext *context)
        : OpConversionPattern<CastToFloatOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CastToFloatOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      FloatType fType = ::llvm::dyn_cast<FloatType>(op.getRes().getType());

      int targetWidth = fType.getWidth();

      RealType fromType =
          ::llvm::dyn_cast<RealType>(op->getOperand(0).getType());

      if (fromType.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion from fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      if (targetWidth > 64) {
        op->emitOpError()
            << "Conversion to floats bigger than f64 is not yet supported";
        return failure();
      }

      if (fromType.getExponent() >
          llvm::APFloat::semanticsMaxExponent(fType.getFloatSemantics())) {
        // maybe add an option for saturating conversion in the future?
        op->emitOpError()
            << "Target float type too small to represent real value";
        return failure();
      }

      auto buildIntAttr = [targetWidth](Builder b,
                                        int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(targetWidth), value);
      };

      // imagine a world where you are a programmer who decides the method
      // "getFPMantissaWidth" returns the width of the mantissa field + 1
      // because it accounts for the implicit bit for some godforsaken reason
      int mantissaBitwidth = fType.getFPMantissaWidth() - 1;

      Value conv =
          (fromType.getSignd())
              ? b.create<arith::SIToFPOp>(fType, adaptor.getFrom()).getResult()
              : b.create<arith::UIToFPOp>(fType, adaptor.getFrom()).getResult();

      // NOTE: there is no need to account for exponent bias here, as it's
      // already implicitly accounted for

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(targetWidth), conv);

      // compute actual exponent in place
      IntegerAttr dtExp = buildIntAttr(
          b, (uint64_t)(std::abs(fromType.getExponent()) << mantissaBitwidth));
      arith::ConstantOp dtExp_const = b.create<arith::ConstantOp>(dtExp);
      Value actual_exp =
          fromType.getExponent() > 0
              ? b.create<arith::AddIOp>(bitcast, dtExp_const).getResult()
              : b.create<arith::SubIOp>(bitcast, dtExp_const).getResult();

      // bitcast back
      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, actual_exp);

      rewriter.replaceOp(op, bitcast_back);
      return success();
    }
  };

  struct ConvertBitcast : public OpConversionPattern<BitcastOp> {
    ConvertBitcast(mlir::MLIRContext *context)
        : OpConversionPattern<BitcastOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      if (op.getFrom().getType() == op.getRes().getType()) {
        op->emitOpError() << "bitcasting to same type";
        return failure();
      }

      rewriter.replaceOp(op, adaptor.getFrom());
      return success();
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::Operation *module = getOperation();

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addIllegalDialect<TaffoDialect>();

    RewritePatternSet patterns(context);
    TaffoToArithTypeConverter typeConverter(context);
    patterns.add<ConvertAdd, ConvertMult, ConvertCastToReal, ConvertCastToFloat,
                 ConvertBitcast>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir