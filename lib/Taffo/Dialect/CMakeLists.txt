add_mlir_dialect_library(MLIRTaffo
        TaffoDialect.cpp
        Ops.cpp
        Types.cpp

        DEPENDS
        MLIRTaffoIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
)
