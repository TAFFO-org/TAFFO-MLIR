#include "Taffo/Transforms/LowerToArithPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
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

            auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
                loc, resultType, inputs);

            return castOp.getResult(0);
          });

      addSourceMaterialization(
          [&](mlir::OpBuilder &builder, mlir::Type resultType,
              mlir::ValueRange inputs,
              mlir::Location loc) -> std::optional<mlir::Value> {
            if (inputs.size() != 1) {
              return std::nullopt;
            }

            auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
                loc, resultType, inputs);

            return castOp.getResult(0);
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
  struct ConvertCast : public OpConversionPattern<CastOp> {
    ConvertCast(mlir::MLIRContext *context)
        : OpConversionPattern<CastOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CastOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      FloatType fType = ::llvm::dyn_cast<FloatType>(op.getFrom().getType());
      unsigned ogWidth = fType.getWidth();

      if (ogWidth > 64) {
        op->emitOpError()
            << "Conversion from floats bigger than f64 is not yet supported";
        return failure();
      }

      // sign mask (0b1000...0)
      IntegerAttr sign_mask = b.getIntegerAttr(b.getIntegerType(ogWidth),
                                               (uint64_t)1 << (ogWidth - 1));
      arith::ConstantOp sign_constOp = b.create<arith::ConstantOp>(sign_mask);

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(ogWidth), op.getFrom());

      // check sign
      arith::AndIOp andOp = b.create<arith::AndIOp>(sign_constOp, bitcast);
      arith::CmpIOp isNegativeOp = b.create<arith::CmpIOp>(
          arith::CmpIPredicate::eq, andOp, sign_constOp);

      // inverse of sign mask (0b01111...1)
      IntegerAttr mask2 = b.getIntegerAttr(b.getIntegerType(ogWidth),
                                           ((uint64_t)1 << (ogWidth - 1)) - 1);
      arith::ConstantOp constOp2 = b.create<arith::ConstantOp>(mask2);
      // zero out sign bit
      arith::AndIOp signless = b.create<arith::AndIOp>(constOp2, bitcast);

      // shift mantissa out
      IntegerAttr shift_amount =
          b.getIntegerAttr(b.getIntegerType(ogWidth),
                           ogWidth - (fType.getFPMantissaWidth() + 1));
      arith::ConstantOp constOp3 = b.create<arith::ConstantOp>(shift_amount);
      arith::ShRUIOp expBits = b.create<arith::ShRUIOp>(signless, constOp3);

      // exponent bias (for f32, this is 2^(8 - 1) - 1 = 127)
      IntegerAttr mask4 = b.getIntegerAttr(
          b.getIntegerType(ogWidth),
          ((uint64_t)1 << (fType.getFPMantissaWidth() - 1)) - 1);
      arith::ConstantOp constOp4 = b.create<arith::ConstantOp>(mask4);
      // debias exponent
      arith::SubIOp debiasedExp = b.create<arith::SubIOp>(expBits, constOp4);

      // shift exponent out
      // we leave one bit to add in the implicit 1 to the significand
      IntegerAttr shift_amount2 = b.getIntegerAttr(b.getIntegerType(ogWidth),
                                                   fType.getFPMantissaWidth());
      arith::ConstantOp constOp5 = b.create<arith::ConstantOp>(shift_amount2);
      arith::ShLIOp mantissa = b.create<arith::ShLIOp>(signless, constOp5);

      // add implicit one
      arith::OrIOp decoded_mantissa =
          b.create<arith::OrIOp>(mantissa, sign_constOp);

      // compute shift amount to convert to fixed point
      DatatypeInfoAttr dtInfo =
          op->getAttr("DatatypeInfo").dyn_cast_or_null<DatatypeInfoAttr>();

      IntegerAttr fixP_exp =
          b.getIntegerAttr(b.getIntegerType(ogWidth), dtInfo.getExponent());
      arith::ConstantOp constOp6 = b.create<arith::ConstantOp>(fixP_exp);
      arith::SubIOp shift_amount3 =
          b.create<arith::SubIOp>(constOp6, debiasedExp);

      // final shift
      arith::ShRUIOp fixP =
          b.create<arith::ShRUIOp>(decoded_mantissa, shift_amount3);

      //NOTE: If shift_amount3 == 0 and isNegativeOp == 1 the number is
      // already converted, otherwise we need to do 2's complement
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::Operation *module = getOperation();

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addIllegalOp<AddOp>();
    // target.addIllegalDialect<TaffoDialect>();

    RewritePatternSet patterns(context);
    TaffoToArithTypeConverter typeConverter(context, 32);
    patterns.add<ConvertAdd>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir