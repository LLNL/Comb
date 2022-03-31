##############################################################################
## Copyright (c) 2018-2022, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory
##
## LLNL-CODE-758885
##
## All rights reserved.
##
## This file is part of Comb.
##
## For details, see https://github.com/LLNL/Comb
## Please also see the LICENSE file for MIT license.
##############################################################################

cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

if (APPLE)
 cmake_policy(SET CMP0025 NEW)
endif()

include(CMakeDependentOption)

set(COMB_VERSION_MAJOR 0)
set(COMB_VERSION_MINOR 3)
set(COMB_VERSION_PATCHLEVEL 1)

set(COMB_LOADED "${COMB_VERSION_MAJOR}.${COMB_VERSION_MINOR}.${COMB_VERSION_PATCHLEVEL}")

project(COMB LANGUAGES CXX C VERSION ${COMB_LOADED})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(cmake/SetupCombOptions.cmake)

cmake_minimum_required(VERSION 3.14.5)

# Detect C++ standard and add appropriate flag _before_ loading BLT
set(COMPILERS_KNOWN_TO_CMAKE33 AppleClang Clang GNU MSVC)

include(CheckCXXCompilerFlag)
if(NOT DEFINED BLT_CXX_STD)
  if("cxx_std_17" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    set(BLT_CXX_STD c++17 CACHE STRING "Version of C++ standard")
    message("Using C++ standard: ${BLT_CXX_STD}")
  elseif("cxx_std_14" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    set(BLT_CXX_STD c++14 CACHE STRING "Version of C++ standard")
    message("Using C++ standard: ${BLT_CXX_STD}")
  elseif("${CMAKE_CXX_COMPILER_ID}" IN_LIST COMPILERS_KNOWN_TO_CMAKE33)
    set(BLT_CXX_STD c++14 CACHE STRING "Version of C++ standard")
    message("Using C++ standard: ${BLT_CXX_STD}")
  else() #cmake has no idea what to do, do it ourselves...
    foreach(flag_var "c++17" "c++14")
      CHECK_CXX_COMPILER_FLAG("-std=${flag_var}" COMPILER_SUPPORTS_${flag_var})
      if(COMPILER_SUPPORTS_${flag_var})
        set(BLT_CXX_STD ${flag_var} CACHE STRING "Version of C++ standard")
        message("Using C++ standard: ${BLT_CXX_STD}")
        break()
      endif()
    endforeach(flag_var)
  endif()
else() #check BLT_CXX_STD is high enough by disallowing the only invalid option
  if("${BLT_CXX_STD}" IN_LIST "c++98;c++11")
    message(FATAL_ERROR "RAJA requires minimum C++ standard of c++14")
  endif()
endif(NOT DEFINED BLT_CXX_STD)

set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT BLT_LOADED)
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "\
      The BLT submodule is not present. \
      If in git repository run the following two commands:\n \
      git submodule init\n \
      git submodule update")
    endif ()
  endif ()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

# Setup options that depend on BLT
include(cmake/SetupDependentOptions.cmake)
# Setup basic CMake options
include(cmake/SetupBasics.cmake)
# Find third-party packages
include(cmake/SetupPackages.cmake)
# Setup vendor-specific compiler flags
include(cmake/SetupCompilers.cmake)
# Setup internal COMB configuration options
include(cmake/SetupCombConfig.cmake)

add_subdirectory(src)
