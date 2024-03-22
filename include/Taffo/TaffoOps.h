
#ifndef TAFFO_TAFFOOPS_H
#define TAFFO_TAFFOOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Taffo/TaffoOps.h.inc"

#endif // TAFFO_TAFFOOPS_H
