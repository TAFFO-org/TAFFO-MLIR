# -*- Python -*-

import os
import sys

import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# Name of the test suite.
config.name = 'TAFFO'

# List of file extensions to treat as test files.
config.suffixes = [
    ".mlir"
]

# Test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# Root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Root path where tests should be run.
config.test_exec_root = os.path.join(config.taffo_obj_root, 'test')

# On MacOS, set the environment variable for the path of the SDK to be used.
lit.util.usePlatformSdkOnDarwin(config, lit_config)

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

# Copy system environment.
llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"], append_path=True)
llvm_config.use_default_substitutions()

# List of directories to exclude from the testsuite.
config.excludes = [
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "lit.site.cfg.py"
]

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.taffo_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.taffo_tools_dir,
    config.llvm_tools_dir
]

tools = [
    "taffo-opt"
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Set the LD_LIBRARY_PATH
ld_library_path = os.path.pathsep.join((
    config.llvm_libs_dir,
    config.environment.get("LD_LIBRARY_PATH", "")))

config.environment["LD_LIBRARY_PATH"] = ld_library_path
