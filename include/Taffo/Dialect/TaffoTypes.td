#ifndef TAFFO_DIALECT_TAFFOTYPES_TD
#define TAFFO_DIALECT_TAFFOTYPES_TD

include "Taffo/Dialect/Taffo.td"
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

def Taffo_Real : Taffo_Type<"Real", "real"> {
    let summary = "Real type, can represent any number";
    let description = [{
        real type description.
    }];

    //let parameters = (ins APFloatParameter<"precision">:$precision,
    //                      APFloatParameter<"min">:$min,
    //                      APFloatParameter<"max">:$max);
//
    ////TODO: fix assembly format
    //let assemblyFormat = [{
    //  `<` $precision `,` $min `,` $max `>`
    //}];
    ////let assemblyFormat = [{value `p` `:` $precision `range` `:` `[`$min`,` $max`]`}];

    let extraClassDeclaration = [{
        static }] # name # [{Type get(MLIRContext *context);
    }];
}

#endif // TAFFO_DIALECT_TAFFOTYPES_TD