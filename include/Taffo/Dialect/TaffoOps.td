#ifndef TAFFO_DIALECT_TAFFOOPS_TD
#define TAFFO_DIALECT_TAFFOOPS_TD

include "Taffo/Dialect/Taffo.td"
include "Taffo/Dialect/TaffoTypes.td"
include "Taffo/Interfaces/InferTaffoRangeNtvInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "mlir/Interfaces/InferTypeOpInterface.td"

//===---------------------------------------------------------------------===//
// Type contraints.
//===---------------------------------------------------------------------===//

// Real type.
def TaffoReal :
    Type<CPred<"$_self.isa<::mlir::taffo::RealType>()">,
         "real", "::mlir::taffo::RealType">;

//===---------------------------------------------------------------------===//
// Taffo operation definition.
//===---------------------------------------------------------------------===//

class Taffo_Op<string mnemonic, list<Trait> traits = []> :
    Op<TaffoDialect, mnemonic, traits>;

def Taffo_CastOp : Taffo_Op<"cast",
    [DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo assignment summary";
    let description = [{
        taffo assignment description
    }];

    let arguments = (ins FloatLike:$from,
                         F64Attr:$precision,
                         F64Attr:$min,
                         F64Attr:$max);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$from `,` $precision `,` $min `,` $max attr-dict `:`"
                               "type($from) `->`  type($res)";
}

def Taffo_ConvertOp : Taffo_Op<"convert",
    [DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo convert summary";
    let description = [{
        taffo convert description
    }];

    let arguments = (ins TaffoReal:$from);
    let results = (outs FloatLike:$res);

    let assemblyFormat = "$from attr-dict `:`"
                               "type($from) `->`  type($res)";
}

def Taffo_AddOp : Taffo_Op<"add",
    [SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo add summary";
    let description = [{
        taffo add description
    }];

    let arguments = (ins TaffoReal:$lhs, TaffoReal:$rhs);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

#endif // TAFFO_DIALECT_TAFFOOPS_TD