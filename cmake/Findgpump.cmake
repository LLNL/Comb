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

find_path(GPUMP_PATH
    NAMES "lib/libgpump.so"
    PATHS
      ENV GPUMP_DIR
      /opt/ibm/spectrum_mpi/libgpump
    DOC "Path to gpump library")


if(GPUMP_PATH)
    message(STATUS "GPUMP_PATH:  ${GPUMP_PATH}")
    set(GPUMP_FOUND TRUE)
    set(GPUMP_CXX_COMPILE_FLAGS -I${GPUMP_PATH}/include)
    set(GPUMP_INCLUDE_PATH      ${GPUMP_PATH}/include)
    set(GPUMP_CXX_LINK_FLAGS    -L${GPUMP_PATH}/lib)
    set(GPUMP_CXX_LIBRARIES     ${GPUMP_PATH}/lib/libgpump.so)
    set(GPUMP_ARCH              )
else()
    set(GPUMP_FOUND FALSE)
    message(WARNING "gpump library not found")
endif()
