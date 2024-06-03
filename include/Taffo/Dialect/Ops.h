#ifndef TAFFO_DIALECT_OPS_H
#define TAFFO_DIALECT_OPS_H

#include "Taffo/Interfaces/InferTaffoRangeNtvInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//possible duplicates from marco
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"

#include "llvm/ADT/APFloat.h"

#include "Taffo/Dialect/Types.h"

#define GET_OP_FWD_DEFINES
#include "Taffo/Dialect/Taffo.h.inc"

#define GET_OP_CLASSES
#include "Taffo/Dialect/Taffo.h.inc"

#endif // TAFFO_DIALECT_OPS_H
