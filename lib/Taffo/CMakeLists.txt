add_mlir_dialect_library(MLIRTaffo
        TaffoDialect.cpp
        TaffoOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Taffo

        DEPENDS
        MLIRTaffoOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)
