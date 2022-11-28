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

###############################################################################
#
# Setup rocTX
# This file defines:
#  ROCTX_FOUND - If rocTX was found
#  ROCTX_INCLUDE_DIRS - The rocTX include directories
#  ROCTX_LIBRARY - The rocTX library

#find includes
find_path( ROCTX_INCLUDE_DIRS
  NAMES roctx.h
  HINTS
    ${ROCTX_DIR}/include
    ${ROCTRACER_DIR}/include
    ${HIP_ROOT_DIR}/../roctracer/include
    ${HIP_ROOT_DIR}/../include )

find_library( ROCTX_LIBRARY
  NAMES roctx64 libroctx64
  HINTS
    ${ROCTX_DIR}/lib
    ${ROCTRACER_DIR}/lib
    ${HIP_ROOT_DIR}/../roctracer/lib
    ${HIP_ROOT_DIR}/../lib )


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ROCTX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ROCTX  DEFAULT_MSG
                                  ROCTX_INCLUDE_DIRS
                                  ROCTX_LIBRARY )
