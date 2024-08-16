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
    TaffoToArithTypeConverter(MLIRContext *ctx, int bitwidth) {
      addConversion([](Type type) { return type; });
      addConversion([ctx, bitwidth](RealType type) -> Type {
        return IntegerType::get(ctx, bitwidth,
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

  struct ConvertAdd : public OpConversionPattern<AddOp> {
    ConvertAdd(mlir::MLIRContext *context)
        : OpConversionPattern<AddOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    // TODO overflow check (maybe in an intermediate pass on dtInfo?)
    LogicalResult
    matchAndRewrite(AddOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      // TODO handle function arguments
      auto getExp = [](Value v) -> std::optional<int> {
        mlir::Operation *op = v.getDefiningOp();
        if (op == nullptr)
          return std::nullopt;
        Attribute attr = op->getAttr("DatatypeInfo");
        DatatypeInfoAttr dt = ::llvm::dyn_cast_or_null<DatatypeInfoAttr>(attr);
        return dt ? std::optional<int>{dt.getExponent()} : std::nullopt;
      };

      std::optional<int> rhsExp = getExp(op.getRhs());
      if (!rhsExp) {
        op->emitOpError() << "DatatypeInfo has not been set for rhs";
        return failure();
      }
      std::optional<int> lhsExp = getExp(op.getLhs());
      if (!lhsExp) {
        op->emitOpError() << "DatatypeInfo has not been set for lhs";
        return failure();
      }

      int expDiff = std::abs(rhsExp.value() - lhsExp.value());
      if (expDiff > 32) {
        // TODO if difference between exps is greater than bitwidth, delete
        //  op with warning
      }
      if (expDiff != 0) {
        Value to_shift = rhsExp.value() < lhsExp.value() ? adaptor.getRhs()
                                                         : adaptor.getLhs();
        Value no_shift = rhsExp.value() > lhsExp.value() ? adaptor.getRhs()
                                                         : adaptor.getLhs();

        arith::ConstantOp shift_amount =
            b.create<arith::ConstantOp>(b.getI32IntegerAttr(expDiff));
        arith::ShRSIOp ShOp =
            b.create<arith::ShRSIOp>(to_shift, shift_amount.getResult());
        arith::AddIOp addOp =
            b.create<arith::AddIOp>(no_shift, ShOp.getResult());
        rewriter.replaceOp(op, addOp);
        return success();
      }

      arith::AddIOp addOp =
          b.create<arith::AddIOp>(adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, addOp);
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

      DatatypeInfoAttr dtInfo =
          op->getAttr("DatatypeInfo").dyn_cast_or_null<DatatypeInfoAttr>();

      if (dtInfo.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion to fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      auto buildIntAttr = [ogWidth](Builder b, int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(ogWidth), value);
      };

      auto buildDestIntAttr = [dtInfo](Builder b,
                                       int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(dtInfo.getBitwidth()), value);
      };

      // imagine a world where you are a programmer who decides the method
      // "getFPMantissaWidth" returns the width of the mantissa field + 1
      // because it accounts for the implicit bit for some godforsaken reason
      int mantissaBitwidth = fType.getFPMantissaWidth() - 1;

      int floatExpBitwidth = ogWidth - mantissaBitwidth - 1;

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(ogWidth), op.getFrom());

      // zero out exponent
      uint64_t exponent_mask =
          // 0b1000...0
          ((uint64_t)1 << (ogWidth - 1)) +
          // 0b((0)^exp_size+1) 1111...1
          ((uint64_t)1 << (mantissaBitwidth)) - 1;
      arith::ConstantOp exp_mask_const =
          b.create<arith::ConstantOp>(buildIntAttr(b, exponent_mask));
      arith::AndIOp no_exp = b.create<arith::AndIOp>(bitcast, exp_mask_const);

      // set exponent (for f32, we set to 30 such that we have room for the sign
      // bit)
      uint64_t bias = ((uint64_t)1 << (floatExpBitwidth - 1)) - 1;
      uint64_t new_exp = ((uint64_t)(bias + dtInfo.getBitwidth() - 2))
                         << (mantissaBitwidth);
      arith::ConstantOp new_exp_const =
          b.create<arith::ConstantOp>(buildIntAttr(b, new_exp));
      arith::OrIOp normalized = b.create<arith::OrIOp>(no_exp, new_exp_const);

      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, normalized);

      arith::FPToSIOp conv = b.create<arith::FPToSIOp>(
          b.getIntegerType(dtInfo.getBitwidth()), bitcast_back);

      // inverse of sign mask (0b01111...1)
      IntegerAttr inv_sign_mask =
          buildIntAttr(b, ((uint64_t)1 << (ogWidth - 1)) - 1);
      arith::ConstantOp inv_sign_mask_const =
          b.create<arith::ConstantOp>(inv_sign_mask);
      // zero out sign bit
      arith::AndIOp signless =
          b.create<arith::AndIOp>(bitcast, inv_sign_mask_const);

      // shift mantissa out
      IntegerAttr shift_amount = buildIntAttr(b, mantissaBitwidth);
      arith::ConstantOp shift_amount_const =
          b.create<arith::ConstantOp>(shift_amount);
      arith::ShRUIOp expBits =
          b.create<arith::ShRUIOp>(signless, shift_amount_const);

      // TODO: final_shift_amount is not quite right. It's not just the
      //  difference between exponents, the bitwidth is somehow related to it

      // account for bias here to save one instruction
      IntegerAttr fixp_exp = buildIntAttr(b, dtInfo.getExponent() + bias);
      arith::ConstantOp fixp_exp_const = b.create<arith::ConstantOp>(fixp_exp);
      arith::SubIOp final_shift_amount =
          // these operands really should not be flipped, I don't get it
          b.create<arith::SubIOp>(expBits, fixp_exp_const);

      int smallestExp =
          llvm::APFloat::semanticsMinExponent(fType.getFloatSemantics());

      int expDiff = dtInfo.getExpDiff()
                        ? dtInfo.getExpDiff().value()
                        : std::abs(dtInfo.getExponent() - smallestExp);

      // if the following is true condition is true, we might shift by a
      // number larger than the bitwidth, which will produce poison, so we
      // need to check and return zero if that is the case
      if (expDiff >= dtInfo.getBitwidth()) {
        arith::ConstantOp dtBitwidth_const =
            b.create<arith::ConstantOp>(buildIntAttr(b, dtInfo.getBitwidth()));
        arith::CmpIOp oob_shift = b.create<arith::CmpIOp>(
            arith::CmpIPredicate::uge, final_shift_amount, dtBitwidth_const);

        Value fsa = final_shift_amount;
        if (ogWidth != dtInfo.getBitwidth()) {
          fsa = ogWidth > dtInfo.getBitwidth()
                    ? b.create<arith::TruncIOp>(
                           b.getIntegerType(dtInfo.getBitwidth()), fsa)
                          .getResult()
                    : b.create<arith::ExtUIOp>(b.getIntegerType(ogWidth), fsa)
                          .getResult();
        }

        arith::ShRSIOp res = b.create<arith::ShRSIOp>(conv.getResult(), fsa);

        arith::ConstantOp zero_const =
            b.create<arith::ConstantOp>(buildDestIntAttr(b, 0));
        arith::SelectOp check_shift =
            b.create<arith::SelectOp>(oob_shift, zero_const, res);

        rewriter.replaceOp(op, check_shift);
        return success();
      }

      Value fsa = final_shift_amount;
      if (ogWidth != dtInfo.getBitwidth()) {
        fsa = ogWidth > dtInfo.getBitwidth()
                  ? b.create<arith::TruncIOp>(
                         b.getIntegerType(dtInfo.getBitwidth()), fsa)
                        .getResult()
                  : b.create<arith::ExtUIOp>(b.getIntegerType(ogWidth), fsa)
                        .getResult();
      }
      arith::ShRSIOp res = b.create<arith::ShRSIOp>(conv, fsa);
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

      DatatypeInfoAttr dtInfo =
          op->getAttr("DatatypeInfo").dyn_cast_or_null<DatatypeInfoAttr>();

      if (dtInfo.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion from fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      if (targetWidth > 64) {
        op->emitOpError()
            << "Conversion to floats bigger than f64 is not yet supported";
        return failure();
      }

      if (dtInfo.getExponent() >
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

      Value conv;
      if (dtInfo.getSignd()) {
        conv = b.create<arith::SIToFPOp>(fType, adaptor.getFrom()).getResult();
      } else {
        conv = b.create<arith::UIToFPOp>(fType, adaptor.getFrom()).getResult();
      }

      // first we extract the exponent

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(targetWidth), conv);

      // inverse of sign mask (0b01111...1)
      IntegerAttr inv_sign_mask =
          buildIntAttr(b, ((uint64_t)1 << (targetWidth - 1)) - 1);
      arith::ConstantOp inv_sign_mask_const =
          b.create<arith::ConstantOp>(inv_sign_mask);
      // zero out sign bit
      arith::AndIOp signless =
          b.create<arith::AndIOp>(bitcast, inv_sign_mask_const);

      // shift mantissa out
      IntegerAttr shift_amount =
          buildIntAttr(b, targetWidth - (mantissaBitwidth + 1));
      arith::ConstantOp shift_amount_const =
          b.create<arith::ConstantOp>(shift_amount);
      arith::ShRUIOp expBits =
          b.create<arith::ShRUIOp>(signless, shift_amount_const);

      // exponent bias (for f32, this is 2^(8 - 1) - 1 = 127)
      IntegerAttr bias =
          buildIntAttr(b, ((uint64_t)1 << (mantissaBitwidth - 1)) - 1);
      arith::ConstantOp bias_const = b.create<arith::ConstantOp>(bias);
      // debias exponent
      arith::SubIOp debiasedExp = b.create<arith::SubIOp>(expBits, bias_const);

      // compute actual exponent
      IntegerAttr dtExp = buildIntAttr(b, dtInfo.getExponent());
      arith::ConstantOp dtExp_const = b.create<arith::ConstantOp>(dtExp);
      arith::AddIOp actual_exp =
          b.create<arith::AddIOp>(dtExp_const, debiasedExp);

      // add bias
      arith::AddIOp biased_actual_exp =
          b.create<arith::AddIOp>(actual_exp, bias_const);

      // shift into correct bits
      IntegerAttr n_mantissa_bits = buildIntAttr(b, mantissaBitwidth);
      arith::ConstantOp n_mantissa_bits_const =
          b.create<arith::ConstantOp>(n_mantissa_bits);
      arith::ShLIOp final_exp =
          b.create<arith::ShLIOp>(biased_actual_exp, n_mantissa_bits_const);

      // zero out exponent original
      uint64_t exponent_mask =
          // 0b1000...0
          ((uint64_t)1 << (targetWidth - 1)) +
          // 0b((0)^exp_size+1) 1111...1
          ((uint64_t)-1 >> (mantissaBitwidth + 1));
      arith::ConstantOp exp_mask_const =
          b.create<arith::ConstantOp>(buildIntAttr(b, exponent_mask));
      arith::AndIOp no_exp = b.create<arith::AndIOp>(bitcast, exp_mask_const);

      // set exponent
      arith::OrIOp final_value = b.create<arith::OrIOp>(no_exp, final_exp);

      // bitcast back
      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, final_value);

      rewriter.replaceOp(op, bitcast_back);
      return success();
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::Operation *module = getOperation();

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addIllegalOp<AddOp, CastToRealOp, CastToFloatOp>();
    target.addIllegalDialect<TaffoDialect>();

    RewritePatternSet patterns(context);
    TaffoToArithTypeConverter typeConverter(context, 32);
    patterns.add<ConvertAdd, ConvertCastToReal, ConvertCastToFloat>(
        typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir