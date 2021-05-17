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

set(COMB_COMPILER "COMB_COMPILER_PGI" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fast -mp" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -fast -mp" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -mp" CACHE STRING "")

set(COMB_HOST_CONFIG_LOADED On CACHE BOOL "")
