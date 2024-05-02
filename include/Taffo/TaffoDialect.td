#ifndef TAFFO_DIALECT
#define TAFFO_DIALECT

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
}

#endif // TAFFO_DIALECT
