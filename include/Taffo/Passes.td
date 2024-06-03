#ifndef TAFFO_PASSES_TD
#define TAFFO_PASSES_TD

include "mlir/Pass/PassBase.td"

def ValueRangeAnalysisPass
    : Pass<"value-range-analysis">
{
    let summary = "Infer the value range of every SSA value.";

    let description = [{
        Infer the value range of every SSA value.
        In the future this will rely on a yet to be implemented trip count
        estimation pass, and will be used for data type allocation.
    }];

    let dependentDialects = [
        "mlir::taffo::TaffoDialect",
    ];

}


#endif TAFFO_PASSES_TD