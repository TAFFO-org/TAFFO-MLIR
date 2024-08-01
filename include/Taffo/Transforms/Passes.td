#ifndef TAFFO_TRANSFORMS_PASSES_TD
#define TAFFO_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ValueRangeAnalysisPass
    : Pass<"value-range-analysis">
{
    let summary = "Infer the value range of every SSA value.";

    let description = [{
        Infer the value range of every SSA value.
        In the future this will rely on a yet to be implemented trip count
        estimation pass, and will be used for data type allocation.

        Currently only works with interval arithmetic, but it's planned to
        use affine arithmetic as well in the future.

        Attaches a "DatatypeInfo" attribute to each op to be used by other
        passes.
    }];

    let dependentDialects = [
        "mlir::taffo::TaffoDialect"
    ];

}

def LowerToArithPass
    : Pass<"lower-to-arith">
{
    let summary = "Lower Taffo ops to Arith dialect.";

    let description = [{
        Lower Taffo ops to Arith dialect.
        Depends on ValueRangeAnalysisPass and will fail if not correctly
        sequenced with it.
        Currently only the following ops are supported:
        add, mult, cast (float to fixp) and convert (fixp to float)
    }];

    let dependentDialects = [
        "mlir::taffo::TaffoDialect",
        "mlir::arith::ArithDialect"
    ];

}


#endif // TAFFO_TRANSFORMS_PASSES_TD