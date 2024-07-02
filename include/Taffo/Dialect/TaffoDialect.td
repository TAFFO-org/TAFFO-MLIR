#ifndef TAFFO_DIALECT_TAFFODIALECT
#define TAFFO_DIALECT_TAFFODIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Taffo dialect definition.
//===----------------------------------------------------------------------===//

def TaffoDialect : Dialect {
    let name = "taffo";
    let summary = "An MLIR dialect for precision tuning.";
    let description = [{
        description here.
    }];
    let cppNamespace = "::mlir::taffo";

    let extraClassDeclaration = [{
        void registerTypes();
    }];

    let useDefaultTypePrinterParser = 1;
}

#endif // TAFFO_DIALECT_TAFFODIALECT
