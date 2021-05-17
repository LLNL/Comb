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

find_path(MP_PATH
    NAMES "lib/libmp.so"
    PATHS
      ENV MP_DIR
      /opt/ibm/spectrum_mpi/libmp
    DOC "Path to mp library")


if(MP_PATH)
    message(STATUS "MP_PATH:  ${MP_PATH}")
    set(MP_FOUND TRUE)
    set(MP_CXX_COMPILE_FLAGS -I${MP_PATH}/include)
    set(MP_INCLUDE_PATH      ${MP_PATH}/include)
    set(MP_CXX_LINK_FLAGS    -L${MP_PATH}/lib)
    set(MP_CXX_LIBRARIES     ${MP_PATH}/lib/libmp.so)
    set(MP_ARCH              )
else()
    set(MP_FOUND FALSE)
    message(WARNING "mp library not found")
endif()
