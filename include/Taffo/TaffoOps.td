#ifndef TAFFO_OPS
#define TAFFO_OPS

include "TaffoDialect.td"

def Taffo_AddOp : Taffo_Op<"Add"> {
    let summary = "taffo add summary";
    let description = [{
        taffo add description
    }];

    let arguments = (ins Taffo_Real:$lhs, Taffo_Real:$rhs);
    let results = (outs Taffo_Real:$res);
}

#endif // TAFFO_OPS