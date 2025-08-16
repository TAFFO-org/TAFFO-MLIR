cmake_minimum_required(VERSION 3.13)

project(libaffine_compare_tests LANGUAGES CXX)

find_package(Boost REQUIRED)

add_executable(libaffine_compare
  compare_affine_vs_boost.cpp
)

target_include_directories(libaffine_compare PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Link to project libraries that provide libaffine and LLVM symbols
target_link_libraries(libaffine_compare PRIVATE MLIRTaffoTransforms MLIRTaffoInterfaces)

set_target_properties(libaffine_compare PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
