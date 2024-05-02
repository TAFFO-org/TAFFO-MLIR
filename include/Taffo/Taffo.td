#ifndef TAFFO_TAFFO_TD
#define TAFFO_TAFFO_TD

include "Taffo/TaffoDialect.td"
include "Taffo/TaffoTypes.td"

//possible duplicates from marco
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
//include "mlir/Interfaces/CallInterfaces.td"
//include "mlir/Interfaces/ControlFlowInterfaces.td"
//include "mlir/Interfaces/FunctionInterfaces.td"
//include "mlir/Interfaces/InferTypeOpInterface.td"
//include "mlir/Interfaces/LoopLikeInterface.td"
//include "mlir/Interfaces/SideEffectInterfaces.td"
//include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"


//===----------------------------------------------------------------------===//
// Base taffo operation definition.
//===----------------------------------------------------------------------===//

class Taffo_Op<string mnemonic, list<Trait> traits = []> :
    Op<TaffoDialect, mnemonic, traits>;


def Taffo_AddOp : Taffo_Op<"Add"> {
    let summary = "taffo add summary";
    let description = [{
        taffo add description
    }];

    let hasCustomAssemblyFormat = 0;
    let arguments = (ins Taffo_Real:$lhs, Taffo_Real:$rhs);
    let results = (outs Taffo_Real:$res);
}

#endif // TAFFO_TAFFO_TD