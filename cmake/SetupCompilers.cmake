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

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0" CACHE STRING "")

if (COMB_ENABLE_MODULES AND CMAKE_CXX_COMPILER_ID MATCHES Clang)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmodules")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
    message(FATAL_ERROR "COMB requires GCC 4.9 or greater!")
  endif ()
endif()

set(COMB_COMPILER "COMB_COMPILER_${CMAKE_CXX_COMPILER_ID}")

if ( MSVC )
  if (NOT BUILD_SHARED_LIBS)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
endif()

if (ENABLE_CUDA)

  set(CMAKE_CUDA_STANDARD "14" CACHE STRING "Version of C++ standard for CUDA Builds")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict -arch ${CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr -Xcudafe \"--display_error_number\"")

  if (NOT COMB_HOST_CONFIG_LOADED)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo -O2")
  endif()
endif()
# end COMB_ENABLE_CUDA section

if (ENABLE_HIP)

  set(CMAKE_HIP_STANDARD "14" CACHE STRING "Version of C++ standard for HIP Builds")
  set(CMAKE_HIP_CLANG_FLAGS "${CMAKE_HIP_CLANG_FLAGS} --offload-arch ${HIP_ARCH}")

endif()
# end COMB_ENABLE_HIP section
