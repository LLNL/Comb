##############################################################################
## Copyright (c) 2018-2020, Lawrence Livermore National Security, LLC.
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

set(comb_do_cycles_sources )

macro( builddocycles COMPOL_in EXECMESH_in EXECMANY_in EXECFEW_in)

  set(COMPOL ${COMPOL_in})
  set(EXECMESH ${EXECMESH_in})
  set(EXECMANY ${EXECMANY_in})
  set(EXECFEW ${EXECFEW_in})

  configure_file( do_cycles.cpp.in
                  do_cycles-${COMPOL}-${EXECMESH}-${EXECMANY}-${EXECFEW}.cpp )
  set(comb_do_cycles_sources ${comb_do_cycles_sources} ${CMAKE_CURRENT_BINARY_DIR}/do_cycles-${COMPOL}-${EXECMESH}-${EXECMANY}-${EXECFEW}.cpp)

  unset(EXECFEW)
  unset(EXECMANY)
  unset(EXECMESH)
  unset(COMPOL)

endmacro()

# add variants of do_cycles for the given com policy
macro( builddocyclescom COMPOL_in ENABLE_MPI_TYPE_in )

  builddocycles(${COMPOL_in} seq seq seq)

  if(ENABLE_OPENMP)
    builddocycles(${COMPOL_in} omp seq seq)
    builddocycles(${COMPOL_in} omp omp seq)
    builddocycles(${COMPOL_in} omp omp omp)
  endif()

  if (ENABLE_CUDA)
    builddocycles(${COMPOL_in} cuda seq seq)
    if(ENABLE_OPENMP)
      builddocycles(${COMPOL_in} cuda omp seq)
      builddocycles(${COMPOL_in} cuda omp omp)
    endif()
    builddocycles(${COMPOL_in} cuda cuda seq)
    builddocycles(${COMPOL_in} cuda cuda cuda)
    builddocycles(${COMPOL_in} cuda cuda_batch seq)
    builddocycles(${COMPOL_in} cuda cuda_batch cuda_batch)
    builddocycles(${COMPOL_in} cuda cuda_persistent seq)
    builddocycles(${COMPOL_in} cuda cuda_persistent cuda_persistent)
    # builddocycles(${COMPOL_in} cuda cuda_batch_fewgs seq)
    # builddocycles(${COMPOL_in} cuda cuda_batch_fewgs cuda_batch_fewgs)
    # builddocycles(${COMPOL_in} cuda cuda_persistent_fewgs seq)
    # builddocycles(${COMPOL_in} cuda cuda_persistent_fewgs cuda_persistent_fewgs)
    if (COMB_ENABLE_CUDA_GRAPH)
      builddocycles(${COMPOL_in} cuda cuda_graph seq)
      builddocycles(${COMPOL_in} cuda cuda_graph cuda_graph)
    endif()
  endif()

  if(ENABLE_MPI AND ${ENABLE_MPI_TYPE_in})
    builddocycles(${COMPOL_in} seq mpi_type mpi_type)
    if(ENABLE_OPENMP)
      builddocycles(${COMPOL_in} omp mpi_type mpi_type)
    endif()
    if (ENABLE_CUDA)
      builddocycles(${COMPOL_in} cuda mpi_type mpi_type)
    endif()
  endif()

endmacro()

# add per comm policy variants of do_cycles
builddocyclescom(mock On)

if(ENABLE_MPI)
  builddocyclescom(mpi On)
endif()

if (ENABLE_GDSYNC)
  builddocyclescom(gdsync Off)
endif()

if (ENABLE_GPUMP)
  builddocyclescom(gpump Off)
endif()

if (ENABLE_MP)
  builddocyclescom(mp Off)
endif()

if (ENABLE_UMR)
  builddocyclescom(umr Off)
endif()


blt_add_library(
  NAME comb_do_cycles
  SOURCES ${comb_do_cycles_sources}
  DEPENDS_ON ${comb_depends})

if(ENABLE_OPENMP)
  if(ENABLE_CUDA AND (NOT ENABLE_CLANG_CUDA))
    blt_add_target_compile_flags(TO comb_do_cycles FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
  else()
    blt_add_target_compile_flags(TO comb_do_cycles FLAGS ${OpenMP_CXX_FLAGS})
  endif()
  blt_add_target_link_flags(TO comb_do_cycles FLAGS ${OpenMP_CXX_FLAGS})
endif()

target_include_directories(comb_do_cycles
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
