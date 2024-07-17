#define MARCO_DIALECT_TAFFO_TAFFOATTRIBUTES_TD
#ifndef MARCO_DIALECT_TAFFO_TAFFOATTRIBUTES_TD

include "Taffo/Dialect/Taffo.td"
//include "Taffo/Dialect/TaffoAttrInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===---------------------------------------------------------------------===//
// Base Taffo attribute definitions
//===---------------------------------------------------------------------===//

class Taffo_Attr<string name,
                    list<Trait> traits = [],
                    string baseCppClass = "::mlir::Attribute">
    : AttrDef<Taffo_Dialect, name, traits, baseCppClass>
{
    let mnemonic = ?;
}

def Taffo_DatatypeInfoAttr
    : Taffo_Attr<"DatatypeInfo", [TypedAttrInterface]>
{
    let summary = [{
        Taffo attribute for storing information about
        the data type of an operation
                    }];

    let description = [{
        Has 3 parameters:
        - "always positive" bit, for lowering to unsigned intergers
        - exponent
        - bitwidth
    }];

    let mnemonic = "dtInfo";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "bool":$sign,
        "int":$exponent,
        "int":$bitwidth);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type,
                                            "bool":$sign,
                                            "int":$exponent,
                                            "int":$bitwidth), [{
            return $_get(type.getContext(), type, sign, exponent, bitwidth);
        }]>,
        AttrBuilderWithInferredContext<(ins "MLIRContext*":$context, "bool":$value), [{
            return $_get(context, BooleanType::get(context), value);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
}

#endif // MARCO_DIALECT_TAFFO_IR_TAFFOATTRIBUTES_TD