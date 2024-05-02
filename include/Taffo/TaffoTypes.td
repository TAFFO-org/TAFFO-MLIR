#ifndef TAFFOTYPES_TD
#define TAFFOTYPES_TD

include "Taffo/TaffoDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Base taffo type definition.
//===----------------------------------------------------------------------===//

class Taffo_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TaffoDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Taffo_Real : Taffo_Type<"RealProblem", "real"> {
    let summary = "Real type, can represent any number";
    let description = [{
        real type description.
    }];
    let parameters = (ins "APFloat":$precision,
                          "APFloat":$min,
                          "APFloat":$max);

    //TODO: fix assembly format
    let assemblyFormat = [{
      $precision $min $max
    }];
    //let assemblyFormat = [{value `p` `:` $precision `range` `:` `[`$min`,` $max`]`}];

}

#endif // TAFFOTYPES_TD