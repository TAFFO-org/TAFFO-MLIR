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
    Op<Taffo_Dialect, mnemonic, traits>;

def Taffo_CastToRealOp : Taffo_Op<"cast2real",
    [DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo cast2real summary";
    let description = [{
        taffo cast2real description
    }];

    let arguments = (ins FloatLike:$from,
                         F64Attr:$precision,
                         F64Attr:$min,
                         F64Attr:$max);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$from `,` $precision `,` $min `,` $max attr-dict `:`"
                               "type($from) `->`  type($res)";
}

def Taffo_CastToFloatOp : Taffo_Op<"cast2float",
    [DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo cast2real summary";
    let description = [{
        taffo cast2real description
    }];

    let arguments = (ins TaffoReal:$from);
    let results = (outs FloatLike:$res);

    let assemblyFormat = "$from attr-dict `:`"
                               "type($from) `->`  type($res)";
}

def Taffo_BitcastOp : Taffo_Op<"bitcast",
    [DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo bitcast summary";
    let description = [{
        this op is only used for debugging and running tests after the
        dt-optimization pass but before lower-to-arith
    }];

    let arguments = (ins AnyTypeOf<[I32, TaffoReal]>:$from);
    let results = (outs AnyTypeOf<[I32, TaffoReal]>:$res);

    let assemblyFormat = "$from attr-dict `:`"
                               "type($from) `->` type($res)";
}

def Taffo_AddOp : Taffo_Op<"add",
    [DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo add summary";
    let description = [{
        taffo add description
    }];

    let arguments = (ins TaffoReal:$lhs, TaffoReal:$rhs);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:`"
                         "`(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

def Taffo_MultOp : Taffo_Op<"mult",
    [DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo mult summary";
    let description = [{
        taffo mult description
    }];

    let arguments = (ins TaffoReal:$lhs, TaffoReal:$rhs);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:`"
                         "`(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

#endif // TAFFO_DIALECT_TAFFOOPS_TD