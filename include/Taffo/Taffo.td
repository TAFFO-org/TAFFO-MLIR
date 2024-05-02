#ifndef TAFFO_TAFFO_TD
#define TAFFO_TAFFO_TD

include "Taffo/TaffoDialect.td"
include "Taffo/TaffoTypes.td"

//===---------------------------------------------------------------------===//
// Type contraints.
//===---------------------------------------------------------------------===//

// Real type.
def TaffoReal :
    Type<CPred<"$_self.isa<::mlir::taffo::RealProblemType>()">,
         "real", "::mlir::taffo::RealProblemType">;

//===---------------------------------------------------------------------===//
// Taffo operation definition.
//===---------------------------------------------------------------------===//

class Taffo_Op<string mnemonic, list<Trait> traits = []> :
    Op<TaffoDialect, mnemonic, traits>;


def Taffo_AddOp : Taffo_Op<"Add"> {
    let summary = "taffo add summary";
    let description = [{
        taffo add description
    }];

    let hasCustomAssemblyFormat = 0;
    let arguments = (ins TaffoReal:$lhs, TaffoReal:$rhs);
    let results = (outs TaffoReal:$res);
}

#endif // TAFFO_TAFFO_TD