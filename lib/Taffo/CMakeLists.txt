add_mlir_dialect_library(MLIRTaffo
        TaffoDialect.cpp
        Ops.cpp
        Types.cpp
        InferTaffoRangeNtvInterface.cpp
        TaffoRangeCommon.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Taffo

        DEPENDS
        MLIRTaffoIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)
