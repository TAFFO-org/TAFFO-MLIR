add_mlir_dialect_library(MLIRTaffoInterfaces
        InferTaffoRangeNtvInterface.cpp

        DEPENDS
        MLIRTaffoIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)
