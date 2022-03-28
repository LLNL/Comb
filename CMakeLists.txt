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

set(COMB_VERSION_MAJOR 0)
set(COMB_VERSION_MINOR 3)
set(COMB_VERSION_PATCHLEVEL 0)

set(COMB_LOADED "${COMB_VERSION_MAJOR}.${COMB_VERSION_MINOR}.${COMB_VERSION_PATCHLEVEL}")

project(COMB LANGUAGES CXX C VERSION ${COMB_LOADED})

# Build options
set(ENABLE_MPI ON CACHE BOOL "Build MPI support")
set(ENABLE_OPENMP OFF CACHE BOOL "Build OpenMP support")
set(ENABLE_CUDA OFF CACHE BOOL "Build CUDA support")
set(ENABLE_HIP OFF CACHE BOOL "Build HIP support")
set(ENABLE_GDSYNC OFF CACHE BOOL "Build GDSYNC support")
set(ENABLE_GPUMP OFF CACHE BOOL "Build GPUMP support")
set(ENABLE_MP OFF CACHE BOOL "Build MP support")
set(ENABLE_UMR OFF CACHE BOOL "Build UMR support")
set(ENABLE_RAJA ON CACHE BOOL "Build RAJA support")
set(ENABLE_CALIPER OFF CACHE BOOL "Build Caliper support")
set(ENABLE_ADIAK OFF CACHE BOOL "Build Adiak support")

set(COMB_ENABLE_LOG OFF CACHE BOOL "Build logging support")

option(ENABLE_CLANG_CUDA "Use Clang's native CUDA support" OFF)

# Build options for libraries, disable extras
option(ENABLE_TESTS "Build tests" Off)
option(ENABLE_REPRODUCERS "Build issue reproducers" Off)
option(ENABLE_EXAMPLES "Build simple examples" Off)
option(ENABLE_EXERCISES "Build exercises " Off)
option(ENABLE_MODULES "Enable modules in supporting compilers (clang)" Off)


set(COMB_CXX_STANDARD_FLAG "default" CACHE STRING "Specific c++ standard flag to use, default attempts to autodetect the highest available")

cmake_minimum_required(VERSION 3.9)

if (ENABLE_CUDA)
  # Separable compilation is required by comb, set before load BLT
  set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "")
  if (NOT DEFINED CUDA_ARCH)
    message(STATUS "CUDA compute architecture set to Comb default sm_35 since it was not specified")
    set(CUDA_ARCH "sm_35" CACHE STRING "Set CUDA_ARCH to Comb minimum supported" FORCE)
  endif()
endif()

if (ENABLE_HIP)
  # Separable compilation is required by comb, set before load BLT
  set(HIP_SEPARABLE_COMPILATION ON CACHE BOOL "")
  if (NOT DEFINED HIP_ARCH)
    message(STATUS "HIP compute architecture set to Comb default gfx900 since it was not specified")
    set(HIP_ARCH "gfx900" CACHE STRING "Set HIP_ARCH to Comb minimum supported" FORCE)
  endif()
endif()

# Detect C++ standard and add appropriate flag _before_ loading BLT
set(COMPILERS_KNOWN_TO_CMAKE33 AppleClang Clang GNU MSVC)

include(CheckCXXCompilerFlag)
if(COMB_CXX_STANDARD_FLAG MATCHES default)
  if("cxx_std_17" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    #TODO set BLT_CXX_STANDARD
    set(CMAKE_CXX_STANDARD 17)
  elseif("cxx_std_14" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    set(CMAKE_CXX_STANDARD 14)
  elseif("${CMAKE_CXX_COMPILER_ID}" IN_LIST COMPILERS_KNOWN_TO_CMAKE33)
    set(CMAKE_CXX_STANDARD 14)
  else() #cmake has no idea what to do, do it ourselves...
    foreach(flag_var "-std=c++17" "-std=c++1z" "-std=c++14" "-std=c++1y" "-std=c++11")
      CHECK_CXX_COMPILER_FLAG(${flag_var} COMPILER_SUPPORTS_${flag_var})
      if(COMPILER_SUPPORTS_${flag_var})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag_var}")
        break()
      endif()
    endforeach(flag_var)
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMB_CXX_STANDARD_FLAG}")
  message("Using C++ standard flag: ${COMB_CXX_STANDARD_FLAG}")
endif()

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

# Setup basic CMake options
include(cmake/SetupBasics.cmake)
# Find third-party packages
include(cmake/SetupPackages.cmake)
# Setup vendor-specific compiler flags
include(cmake/SetupCompilers.cmake)
# Setup internal COMB configuration options
include(cmake/SetupCombConfig.cmake)

add_subdirectory(src)
