if(NOT LLVM_EXTERNAL_LIT OR NOT EXISTS "${LLVM_EXTERNAL_LIT}")
  set(LLVM_EXTERNAL_LIT
    "${TAFFO_ROOT}/llvm-project/llvm/utils/lit/lit.py"
    CACHE FILEPATH "Path to llvm-lit" FORCE
  )
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TAFFO_TEST_DEPENDS
  taffo-opt
)

add_lit_testsuite(check "Running the regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TAFFO_TEST_DEPENDS}
)

set_target_properties(check PROPERTIES FOLDER "Tests")

add_lit_testsuites(TAFFO ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${TAFFO_TEST_DEPENDS}
)
