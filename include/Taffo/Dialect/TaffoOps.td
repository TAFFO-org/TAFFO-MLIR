#ifndef TAFFO_DIALECT_TAFFOOPS_TD
#define TAFFO_DIALECT_TAFFOOPS_TD

include "Taffo/Dialect/Taffo.td"
include "Taffo/Dialect/TaffoTypes.td"
include "Taffo/Interfaces/InferTaffoRangeInterface.td"
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
    [DeclareOpInterfaceMethods<InferTaffoRangeInterface>]> {
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

    let hasVerifier = 1;
}

def Taffo_CastToFloatOp : Taffo_Op<"cast2float",
    [DeclareOpInterfaceMethods<InferTaffoRangeInterface>]> {
    let summary = "taffo cast2real summary";
    let description = [{
        taffo cast2real description
    }];

    let arguments = (ins TaffoReal:$from);
    let results = (outs FloatLike:$res);

    let assemblyFormat = "$from attr-dict `:`"
                               "type($from) `->`  type($res)";
}

def Taffo_BitcastToRealOp : Taffo_Op<"bitcasti2r"> {
    let summary = "taffo bitcast to real summary";
    let description = [{
        this op is only used for debugging and running tests after the
        dt-optimization pass but before lower-to-arith
    }];

    let arguments = (ins AnySignlessInteger:$from);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$from attr-dict `:`"
                               "type($from) `->` type($res)";
}

def Taffo_BitcastToIntOp : Taffo_Op<"bitcastr2i"> {
    let summary = "taffo bitcast to int summary";
    let description = [{
        this op is only used for debugging and running tests after the
        dt-optimization pass but before lower-to-arith
    }];

    let arguments = (ins  TaffoReal:$from);
    let results = (outs AnySignlessInteger:$res);

    let assemblyFormat = "$from attr-dict `:`"
                               "type($from) `->` type($res)";
}

def Taffo_AddOp : Taffo_Op<"add",
    [DeclareOpInterfaceMethods<InferTaffoRangeInterface>]> {
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
    [DeclareOpInterfaceMethods<InferTaffoRangeInterface>]> {
    let summary = "taffo mult summary";
    let description = [{
        taffo mult description
    }];

    let arguments = (ins TaffoReal:$lhs, TaffoReal:$rhs);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:`"
                         "`(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

def Taffo_SubOp : Taffo_Op<"sub",
    [DeclareOpInterfaceMethods<InferTaffoRangeInterface>]> {
    let summary = "taffo sub summary";
    let description = [{
        taffo sub description
    }];

    let arguments = (ins TaffoReal:$lhs, TaffoReal:$rhs);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:`"
                         "`(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

def Taffo_DivOp : Taffo_Op<"div",
    [DeclareOpInterfaceMethods<InferTaffoRangeInterface>]> {
    let summary = "taffo div summary";
    let description = [{
        taffo div description
    }];

    let arguments = (ins TaffoReal:$lhs, TaffoReal:$rhs);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:`"
                         "`(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

def Taffo_AlignOp : Taffo_Op<"align"> {
    let summary = "aligns TaffoReal types by shifting the underlying integer";

    let description = [{
        This op is only used to decouple OpResults that have multiple users.
        In particular, it's inserted by VRA when loop variant results require it.
        Additionally, it may change the bitwidth of its operand, but this feature
        is not currently used.
    }];

    let arguments = (ins TaffoReal:$from);
    let results = (outs TaffoReal:$res);

    let assemblyFormat = "$from attr-dict `:`"
                                   "type($from) `->` type($res)";

    let hasVerifier = 1;
}

#endif // TAFFO_DIALECT_TAFFOOPS_TD