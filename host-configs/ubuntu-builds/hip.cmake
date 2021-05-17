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

set(HIP_ROOT_DIR "${ROCM_DIR}/hip" CACHE PATH "HIP ROOT directory path")

set(CMAKE_CXX_FLAGS_RELEASE "-O2" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

# set(HIP_COMMON_OPT_FLAGS  "--amdgpu-target=gfx900")
set(HIP_COMMON_OPT_FLAGS )
set(HIP_COMMON_DEBUG_FLAGS)
set(HOST_OPT_FLAGS)

if (ENABLE_OPENMP)
	set(HIP_COMMON_OPT_FLAGS "-fopenmp ${HIP_COMMON_OPT_FLAGS}")
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(COMB_HIPCC_FLAGS "-O2 ${HIP_COMMON_OPT_FLAGS} ${HOST_OPT_FLAGS}" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(COMB_HIPCC_FLAGS "-g -O2 ${HIP_COMMON_OPT_FLAGS} ${HOST_OPT_FLAGS}" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(COMB_HIPCC_FLAGS "-g -O0 ${HIP_COMMON_DEBUG_FLAGS}" CACHE STRING "")
endif()

set(COMB_HOST_CONFIG_LOADED On CACHE BOOL "")
