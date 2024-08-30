#ifndef TAFFO_DIALECT_TAFFOATTRIBUTES_TD
#define TAFFO_DIALECT_TAFFOATTRIBUTES_TD

include "Taffo/Dialect/Taffo.td"
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
    : Taffo_Attr<"DatatypeInfo">
{
    let summary = [{
        Taffo attribute for storing information about
        the data type of an operation
                    }];

    let description = [{
        Has 3 parameters:
        - sign bit, for lowering to unsigned integers (set to 0 when the range
            is >= 0 at both ends)
        - exponent
        - bitwidth
        - difference between smallest and largest exponent. This is optional because
            if the range contains zero, then the smallest exponent depends on the datatype.
            This needs to be tracked because shift operations in arith will produce poison
            if the shift amount is larger than the operand bitwidth.
    }];

    let mnemonic = "dtInfo";

    let parameters = (ins
        "bool":$signd,
        "int":$exponent,
        "int":$bitwidth,
        "std::optional<int>":$expSpan);

    let assemblyFormat = "`<` struct(params) `>`";
    //let assemblyFormat = "`<` $signd `,` $exponent `,` $bitwidth `>`";

    let extraClassDeclaration = [{
        bool operator==(DatatypeInfoAttr other) const {

            bool expSpansNull = (!getExpSpan() && !other.getExpSpan());

            bool expSpansNotNull =
                (getExpSpan() && other.getExpSpan())
                    ? getExpSpan().value() == other.getExpSpan().value()
                    : false;

            return
                getSignd() == other.getSignd()
                && getExponent() == other.getExponent()
                && getBitwidth() == other.getBitwidth()
                && (expSpansNull || expSpansNotNull);
        }
    }];
}

#endif // TAFFO_DIALECT_TAFFOATTRIBUTES_TD