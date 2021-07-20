##############################################################################
## Copyright (c) 2018-2021, Lawrence Livermore National Security, LLC.
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

set(comb_sources
  comb.cpp
  print.cpp
  print_timer.cpp
  warmup.cpp
  test_copy.cpp
  test_cycles_mock.cpp
  test_cycles_mpi.cpp
  test_cycles_gdsync.cpp
  test_cycles_gpump.cpp
  test_cycles_mp.cpp
  test_cycles_umr.cpp
  test_cycles_basic.cpp)

macro( builddocycles COMPOL_in EXECMESH_in EXECMANY_in EXECFEW_in)

  set(COMPOL ${COMPOL_in})
  set(EXECMESH ${EXECMESH_in})
  set(EXECMANY ${EXECMANY_in})
  set(EXECFEW ${EXECFEW_in})

  configure_file( do_cycles.cpp.in
                  do_cycles-${COMPOL}-${EXECMESH}-${EXECMANY}-${EXECFEW}.cpp )
  set(comb_sources ${comb_sources} ${CMAKE_CURRENT_BINARY_DIR}/do_cycles-${COMPOL}-${EXECMESH}-${EXECMANY}-${EXECFEW}.cpp)

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
    if (COMB_ENABLE_CUDA_GRAPH)
      builddocycles(${COMPOL_in} cuda cuda_graph seq)
      builddocycles(${COMPOL_in} cuda cuda_graph cuda_graph)
    endif()
  endif()

  if (ENABLE_RAJA)
    builddocycles(${COMPOL_in} raja_seq raja_seq raja_seq)

    if(ENABLE_OPENMP)
      builddocycles(${COMPOL_in} raja_omp raja_seq raja_seq)
      builddocycles(${COMPOL_in} raja_omp raja_omp raja_seq)
      builddocycles(${COMPOL_in} raja_omp raja_omp raja_omp)
    endif()

    if (ENABLE_CUDA)
      builddocycles(${COMPOL_in} raja_cuda raja_seq raja_seq)
      if(ENABLE_OPENMP)
        builddocycles(${COMPOL_in} raja_cuda raja_omp raja_seq)
        builddocycles(${COMPOL_in} raja_cuda raja_omp raja_omp)
      endif()
      builddocycles(${COMPOL_in} raja_cuda raja_cuda raja_seq)
      builddocycles(${COMPOL_in} raja_cuda raja_cuda raja_cuda)
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
    if (ENABLE_RAJA)
      builddocycles(${COMPOL_in} raja_seq mpi_type mpi_type)
      if(ENABLE_OPENMP)
        builddocycles(${COMPOL_in} raja_omp mpi_type mpi_type)
      endif()
      if (ENABLE_CUDA)
        builddocycles(${COMPOL_in} raja_cuda mpi_type mpi_type)
      endif()
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


set(comb_depends )

if(ENABLE_MPI)
  set(comb_depends ${comb_depends} mpi)
endif()

if(ENABLE_OPENMP)
  set(comb_depends ${comb_depends} openmp)
endif()

if (ENABLE_CUDA)
  set(comb_depends ${comb_depends} cuda nvToolsExt)
endif()

if (ENABLE_GDSYNC)
  set(comb_depends ${comb_depends} gdsync)
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

if (ENABLE_RAJA)
  set(comb_depends ${comb_depends} RAJA)
endif()

if (ENABLE_CALIPER)
  list(APPEND comb_depends Caliper)
endif()

if (ENABLE_ADIAK)
  list(APPEND comb_depends Adiak)
endif()

blt_add_executable(
  NAME comb
  SOURCES ${comb_sources}
  DEPENDS_ON ${comb_depends})

target_include_directories(comb
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS comb
  EXPORT comb
  DESTINATION bin)
