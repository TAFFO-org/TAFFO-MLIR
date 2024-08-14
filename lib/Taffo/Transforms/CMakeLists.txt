add_mlir_dialect_library(MLIRTaffoTransforms
        TaffoRangeCommon.cpp
        NtvRangeAnalysis.cpp
        ValueRangeAnalysisPass.cpp
        LowerToArithPass.cpp
        DatatypeOptimizationPass.cpp

        #ADDITIONAL_HEADER_DIRS
        #${PROJECT_SOURCE_DIR}/include/Taffo

        DEPENDS
        MLIRTaffoTransformsIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRTaffoInterfaces
        MLIRIR
)
