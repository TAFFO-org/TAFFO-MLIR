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

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(ogWidth), op.getFrom());

      // declared outside for scoping reasons
      arith::CmpIOp isNegative;
      Value intermediate = bitcast.getResult();

      if (dtInfo.getSignd()) {
        // sign mask (0b1000...0)
        IntegerAttr sign_mask = buildIntAttr(b, (uint64_t)1 << (ogWidth - 1));
        arith::ConstantOp sign_constOp = b.create<arith::ConstantOp>(sign_mask);

        // check sign
        arith::AndIOp andOp = b.create<arith::AndIOp>(sign_constOp, bitcast);
        isNegative = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, andOp,
                                             sign_constOp);

        // inverse of sign mask (0b01111...1)
        IntegerAttr mask2 = buildIntAttr(b, ((uint64_t)1 << (ogWidth - 1)) - 1);
        arith::ConstantOp constOp2 = b.create<arith::ConstantOp>(mask2);
        // zero out sign bit
        arith::AndIOp signless = b.create<arith::AndIOp>(constOp2, bitcast);

        intermediate = signless.getResult();
      }

      // shift mantissa out
      IntegerAttr shift_amount =
          buildIntAttr(b, ogWidth - (fType.getFPMantissaWidth() + 1));
      arith::ConstantOp constOp3 = b.create<arith::ConstantOp>(shift_amount);
      arith::ShRUIOp expBits = b.create<arith::ShRUIOp>(intermediate, constOp3);

      // exponent bias (for f32, this is 2^(8 - 1) - 1 = 127)
      IntegerAttr mask4 = buildIntAttr(
          b, ((uint64_t)1 << (fType.getFPMantissaWidth() - 1)) - 1);
      arith::ConstantOp constOp4 = b.create<arith::ConstantOp>(mask4);
      // debias exponent
      arith::SubIOp debiasedExp = b.create<arith::SubIOp>(expBits, constOp4);

      // shift exponent out
      // we leave one bit to add in the implicit 1 to the significand
      IntegerAttr shift_amount2 = buildIntAttr(b, fType.getFPMantissaWidth());
      arith::ConstantOp constOp5 = b.create<arith::ConstantOp>(shift_amount2);
      arith::ShLIOp mantissa = b.create<arith::ShLIOp>(intermediate, constOp5);

      // add implicit one
      // sign mask (0b1000...0)
      IntegerAttr sign_mask = buildIntAttr(b, (uint64_t)1 << (ogWidth - 1));
      arith::ConstantOp sign_constOp2 = b.create<arith::ConstantOp>(sign_mask);
      arith::OrIOp decoded_mantissa =
          b.create<arith::OrIOp>(mantissa, sign_constOp2);

      // compute shift amount to convert to fixed point
      IntegerAttr fixP_exp = buildIntAttr(b, dtInfo.getExponent()));
      arith::ConstantOp fixP_exp_const = b.create<arith::ConstantOp>(fixP_exp);
      arith::SubIOp final_shift_amount =
          b.create<arith::SubIOp>(fixP_exp_const, debiasedExp);

      int smallestExp = std::floor(std::log2(
          convertToDouble(APFloat::getSmallest(fType.getSemantics(), false))));

      int expDiff = dtInfo.getExpDiff()
                        ? dtInfo.getExpDiff()
                        : std::abs(dtInfo.getExponent() - smallestExp);

      // if the following is true condition is true, we might shift by a
      // number larger than the bitwidth, which will produce poison, so we
      // need to check and return zero if that is the case
      if (expDiff >= dtInfo.getBitwidth()) {

        AddShiftBoundCheck(b, rewriter, final_shift_amount, dtInfo.getBitwidth())
        IntegerAttr dtBitwidth = buildIntAttr(b, dtInfo.getBitwidth());
        arith::ConstantOp dtBitwidth_const =
            b.create<arith::ConstantOp>(dtBitwidth);
        arith::CmpIOp oob_shift = b.create<arith::CmpIOp>(
            arith::CmpIPredicate::geq, final_shift_amount, dtBitwidth_const);

        scf::IfOp check_shift = b.create<scf::IfOp>(
            oob_shift.getResult(), /*then*/
            [&](OpBuilder &b, Location loc) {
              // return zero
              IntegerAttr zero = buildDestIntAttr(b, 0);
              arith::ConstantOp zero_const = b.create<arith::ConstantOp>(zero);
              b.create<scf::YieldOp>(loc, zero_const.getResult());
            }, /*else*/
            [&](OpBuilder &b, Location loc) {
              // otherwise continue as normal with trunc/ext then shift
            });
        rewriter.replaceOp(op, check_shift);
        return success();
      }

      Value res;
      // float type is wider than fixpoint type
      if (ogWidth >= dtInfo.getBitwidth()) {

        // since arith.trunci truncates the most significant bits, we account
        // for that in this shift
        IntegerAttr shift_by = buildIntAttr(
            b, dtInfo.getExponent() + (ogWidth - dtInfo.getBitwidth()));
        arith::SubIOp final_shift_amount =
            b.create<arith::SubIOp>(shift_by, debiasedExp);
        arith::ConstantOp fixP_exp_const =
            b.create<arith::ConstantOp>(fixP_exp);
        arith::ShRUIOp fixP =
            b.create<arith::ShRUIOp>(decoded_mantissa, shift_amount3);
        arith::TruncIOp final = b.create<arith::TruncIOp>(
            b.getIntegerType(dtInfo.getBitwidth()), fixP);
        res = final;
      } else {
      }

      // final shift
      arith::ShRUIOp fixP =
          b.create<arith::ShRUIOp>(decoded_mantissa, shift_amount3);

      // if the number is unsigned, we are done
      if (!dtInfo.getSignd()) {
        rewriter.replaceOp(op, fixP);
        return success();
      }

      scf::IfOp twos_complement = b.create<scf::IfOp>(
          isNegative.getResult(), /*then*/
          [&](OpBuilder &b, Location loc) {
            // 2's complement conversion
            IntegerAttr ones_mask = buildDestIntAttr(b, -1);
            arith::ConstantOp constOp7 =
                b.create<arith::ConstantOp>(loc, ones_mask);
            arith::XOrIOp unaryNot =
                b.create<arith::XOrIOp>(loc, constOp7, fixP);

            IntegerAttr one = buildDestIntAttr(b, 1);
            arith::ConstantOp constOp8 = b.create<arith::ConstantOp>(loc, one);
            arith::AddIOp complement =
                b.create<arith::AddIOp>(loc, constOp8, unaryNot);

            b.create<scf::YieldOp>(loc, complement.getResult());
          }, /*else*/
          [&](OpBuilder &b, Location loc) {
            b.create<scf::YieldOp>(loc, fixP.getResult());
          });

      rewriter.replaceOp(op, twos_complement);
      return success();
    }

    Value ConvertToNarrower() {}

    Value ConvertToWider() {}

    Operation
    AddShiftBoundCheck(OpBuilder b, ConversionPatternRewriter &rewriter,
                       Value shift_amount, int destinationBitwidth,
                       function_ref<void(OpBuilder &, Location)> elseBlock) {

      IntegerAttr dtBitwidth = b.getIntegerAttr(
          shift_amount.getType(), destinationBitwidth);
      arith::ConstantOp dtBitwidth_const =
          b.create<arith::ConstantOp>(dtBitwidth);
      arith::CmpIOp oob_shift = b.create<arith::CmpIOp>(
          arith::CmpIPredicate::uge, shift_amount, dtBitwidth_const);

      scf::IfOp check_shift = b.create<scf::IfOp>(
          oob_shift.getResult(), /*then*/
          [&](OpBuilder &b, Location loc) {
            // return zero
            IntegerAttr zero = buildDestIntAttr(b, 0);
            arith::ConstantOp zero_const = b.create<arith::ConstantOp>(zero);
            b.create<scf::YieldOp>(loc, zero_const.getResult());
          }, /*else*/
          elseBlock);
    }

    auto TruncAndShiftMantissa() {

    }

    auto ShiftMantissa() {

    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::Operation *module = getOperation();

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addIllegalOp<AddOp, CastOp>();
    // target.addIllegalDialect<TaffoDialect>();

    RewritePatternSet patterns(context);
    TaffoToArithTypeConverter typeConverter(context, 32);
    patterns.add<ConvertAdd, ConvertCast>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir