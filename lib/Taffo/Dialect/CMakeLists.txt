add_mlir_dialect_library(MLIRTaffo
    Attributes.cpp
    OpInterfaces.cpp
    Ops.cpp
    Taffo.cpp
    Types.cpp

    DEPENDS
    MLIRTaffoIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
        MLIRInferTypeOpInterface
)
