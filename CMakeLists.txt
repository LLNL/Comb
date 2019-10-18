##############################################################################
## Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

set(COMB_VERSION_MAJOR 0)
set(COMB_VERSION_MINOR 1)
set(COMB_VERSION_PATCHLEVEL 2)

cmake_minimum_required(VERSION 3.9)

set(COMB_LOADED "${COMB_VERSION_MAJOR}.${COMB_VERSION_MINOR}.${COMB_VERSION_PATCHLEVEL}")

project(COMB LANGUAGES CXX VERSION ${COMB_LOADED})

# Build options
set(ENABLE_MPI ON CACHE Bool "Build MPI support")
set(ENABLE_OPENMP OFF CACHE Bool "Build OpenMP support")
set(ENABLE_CUDA OFF CACHE Bool "Build CUDA support")
set(ENABLE_GPUMP OFF CACHE Bool "Build GPUMP support")
set(ENABLE_MP OFF CACHE Bool "Build MP support")
set(ENABLE_UMR OFF CACHE Bool "Build UMR support")

set(COMB_CXX_STANDARD_FLAG "default" CACHE STRING "Specific c++ standard flag to use, default attempts to autodetect the highest available")

option(ENABLE_CLANG_CUDA "Use Clang's native CUDA support" OFF)
set(CUDA_ARCH "sm_70" CACHE STRING "Compute architecture to pass to CUDA builds")

# Separable compilation is required by comb, set before load BLT
set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "")

# Load BLT
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

set(comb_sources
  src/comb.cpp
  src/MultiBuffer.cpp
  src/batch_launch.cpp
  src/persistent_launch.cpp
  src/graph_launch.cpp
  src/print.cpp
  src/print_timer.cpp
  src/warmup.cpp
  src/test_copy.cpp
  src/test_cycles_mock.cpp
  src/test_cycles_mpi.cpp
  src/test_cycles_gpump.cpp
  src/test_cycles_mp.cpp
  src/test_cycles_umr.cpp
  src/test_cycles_basic.cpp)

set(comb_depends )

if(ENABLE_MPI)
  set(comb_depends ${comb_depends} mpi)
endif()

if(ENABLE_OPENMP)
  # set(comb_depends ${comb_depends} openmp)
endif()

if (ENABLE_CUDA)
  set(comb_depends ${comb_depends} cuda nvToolsExt)
endif()

if (ENABLE_GPUMP)
  set(comb_depends ${comb_depends} gpump)
endif()

if (ENABLE_MP)
  set(comb_depends ${comb_depends} mp)
endif()

if (ENABLE_UMR)
  set(comb_depends ${comb_depends} umr)
endif()

blt_add_executable(
  NAME comb
  SOURCES ${comb_sources}
  DEPENDS_ON ${comb_depends})

if(ENABLE_OPENMP)
  if(ENABLE_CUDA AND (NOT ENABLE_CLANG_CUDA))
    blt_add_target_compile_flags(TO comb FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
  else()
    blt_add_target_compile_flags(TO comb FLAGS ${OpenMP_CXX_FLAGS})
  endif()
  blt_add_target_link_flags(TO comb FLAGS ${OpenMP_CXX_FLAGS})
endif()

install(TARGETS comb
  EXPORT comb
  DESTINATION bin)

target_include_directories(comb
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
