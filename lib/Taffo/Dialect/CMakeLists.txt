add_mlir_dialect_library(MLIRTaffo
        Taffo.cpp
        Ops.cpp
        Types.cpp
        Attributes.cpp

        DEPENDS
        MLIRTaffoIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
)
