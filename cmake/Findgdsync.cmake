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

find_path(GDSYNC_PATH
    NAMES "lib/libgdsync.so"
    PATHS
      ENV GDSYNC_DIR
      /opt/ibm/spectrum_mpi/libgdsync
    DOC "Path to gdsync library")


if(GDSYNC_PATH)
    message(STATUS "GDSYNC_PATH:  ${GDSYNC_PATH}")
    set(GDSYNC_FOUND TRUE)
    set(GDSYNC_CXX_COMPILE_FLAGS -I${GDSYNC_PATH}/include)
    set(GDSYNC_INCLUDE_PATH      ${GDSYNC_PATH}/include)
    set(GDSYNC_CXX_LINK_FLAGS    -L${GDSYNC_PATH}/lib)
    set(GDSYNC_CXX_LIBRARIES     ${GDSYNC_PATH}/lib/libgdsync.so)
    set(GDSYNC_ARCH              )
else()
    set(GDSYNC_FOUND FALSE)
    message(WARNING "gdsync library not found")
endif()
