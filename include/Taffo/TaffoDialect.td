
#ifndef TAFFO_DIALECT
#define TAFFO_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Taffo dialect definition.
//===----------------------------------------------------------------------===//

def Taffo_Dialect : Dialect {
    let name = "taffo";
    let summary = "An MLIR dialect for precision tuning.";
    let description = [{
        description here.
    }];
    let cppNamespace = "::mlir::taffo";
}

//===----------------------------------------------------------------------===//
// Base taffo type definition.
//===----------------------------------------------------------------------===//

class Taffo_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Taffo_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Taffo_Real : Taffo_Type<"Real", "real"> {
    let summary = "Real type, can represent any number";
    let description = [{
        real type description.
    }];
    let parameters = (ins "APFloat":$value,
                          "APFloat":$precision,
                          "APFloat":$min,
                          "APFloat":$max);

    let assemblyFormat = [{value `p` `:` $precision `range` `:` `[`$min`,` $max`]`}];

}

//===----------------------------------------------------------------------===//
// Base taffo operation definition.
//===----------------------------------------------------------------------===//

class Taffo_Op<string mnemonic, list<Trait> traits = []> :
        Op<Taffo_Dialect, mnemonic, traits>;


#endif // TAFFO_DIALECT
