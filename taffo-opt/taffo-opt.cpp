#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "Taffo/Dialect/Taffo.h"
#include "Taffo/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<
      mlir::taffo::TaffoDialect,
      mlir::func::FuncDialect,
      mlir::arith::ArithDialect,
      mlir::math::MathDialect,
      mlir::scf::SCFDialect,
      mlir::memref::MemRefDialect,
      mlir::affine::AffineDialect>();

  mlir::taffo::registerTaffoPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Taffo optimizer driver\n", registry));
}
