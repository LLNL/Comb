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

find_path(UMR_PATH
    NAMES "lib/libumr.so"
    PATHS
      ENV UMR_DIR
      /opt/ibm/spectrum_mpi/libumr
    DOC "Path to umr library")


if(UMR_PATH)
    message(STATUS "UMR_PATH:  ${UMR_PATH}")
    set(UMR_FOUND TRUE)
    set(UMR_CXX_COMPILE_FLAGS -I${UMR_PATH}/include)
    set(UMR_INCLUDE_PATH      ${UMR_PATH}/include)
    set(UMR_CXX_LINK_FLAGS    -L${UMR_PATH}/lib)
    set(UMR_CXX_LIBRARIES     ${UMR_PATH}/lib/libumr.so)
    set(UMR_ARCH              )
else()
    set(UMR_FOUND FALSE)
    message(WARNING "umr library not found")
endif()
