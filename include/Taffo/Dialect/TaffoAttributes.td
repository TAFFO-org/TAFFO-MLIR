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


#endif // TAFFO_DIALECT_TAFFOATTRIBUTES_TD