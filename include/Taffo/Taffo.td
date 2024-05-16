#ifndef TAFFO_TAFFO_TD
#define TAFFO_TAFFO_TD

include "Taffo/TaffoDialect.td"
include "Taffo/TaffoTypes.td"
include "Taffo/InferTaffoRangeNtvInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

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

def Taffo_AssignOp : Taffo_Op<"Assign",
    [DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo assignment summary";
    let description = [{
        taffo assignment description
    }];

    let hasCustomAssemblyFormat = 0;
    let arguments = (ins TypedAttrInterface:$precision,
                         TypedAttrInterface:$min,
                         TypedAttrInterface:$max);
    let results = (outs TaffoReal:$res);
}


def Taffo_AddOp : Taffo_Op<"Add",
    [DeclareOpInterfaceMethods<InferTaffoRangeNtvInterface>]> {
    let summary = "taffo add summary";
    let description = [{
        taffo add description
    }];

    let hasCustomAssemblyFormat = 0;
    let arguments = (ins TaffoReal:$lhs, TaffoReal:$rhs);
    let results = (outs TaffoReal:$res);
}

#endif // TAFFO_TAFFO_TD