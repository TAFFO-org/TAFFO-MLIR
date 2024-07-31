#ifndef TAFFO_DIALECT_TAFFO_TD
#define TAFFO_DIALECT_TAFFO_TD

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

    let useDefaultTypePrinterParser = 1;

    let useDefaultAttributePrinterParser = 1;

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];
}

#endif // TAFFO_DIALECT_TAFFODIALECT
