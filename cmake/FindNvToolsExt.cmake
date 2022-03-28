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
# Setup nvToolsExt
# This file defines:
#  NVTOOLSEXT_FOUND - If nvToolsExt was found
#  NVTOOLSEXT_INCLUDE_DIRS - The nvToolsExt include directories
#  NVTOOLSEXT_LIBRARY - The nvToolsExt library

# first Check for CUDA_TOOLKIT_ROOT_DIR
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    MESSAGE(FATAL_ERROR "Could not find NvToolsExt. NvToolsExt support needs explicit CUDA_TOOLKIT_ROOT_DIR")
endif()

#find includes
find_path( NVTOOLSEXT_INCLUDE_DIRS nvToolsExt.h
           HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include )

find_library( NVTOOLSEXT_LIBRARY NAMES nvToolsExt libnvToolsExt
              HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib )


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NVTOOLSEXT_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(NVTOOLSEXT  DEFAULT_MSG
                                  NVTOOLSEXT_INCLUDE_DIRS
                                  NVTOOLSEXT_LIBRARY )
