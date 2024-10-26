#ifndef TAFFO_INTERFACES_INFERTAFFORANGENTVINTERFACE_TD
#define TAFFO_INTERFACES_INFERTAFFORANGENTVINTERFACE_TD

include "Taffo/Dialect/Taffo.td"
include "mlir/IR/OpBase.td"

def InferTaffoRangeNtvInterface : OpInterface<"InferTaffoRangeNtvInterface"> {
  let description = [{
    Allows operations to participate in range analysis for floating point values
  }];
  let cppNamespace = "::mlir::taffo";


let methods = [
  InterfaceMethod<[{
    Infer the bounds on the results of this op given the bounds on its arguments.
    }],
    "void", "inferTaffoRanges", (ins
    "::llvm::ArrayRef<std::pair<::llvm::APFloat,::llvm::APFloat>>":$argRanges,
    "::mlir::taffo::SetTaffoRangeFn":$setResultRanges)
  >,
  
  InterfaceMethod<[{
    Infer the bounds on the results of this op given the bounds on its arguments.
    }],
    "void", "inferTaffoAffineRanges", (ins
    "::llvm::ArrayRef<LibAffine::Var>":$argRanges,
    "::mlir::taffo::SetTaffoAffineRangeFn":$setResultRanges)
  >];
}

#endif // TAFFO_INTERFACES_INFERTAFFORANGENTVINTERFACE_TD