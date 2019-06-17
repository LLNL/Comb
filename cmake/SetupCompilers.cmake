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

set(COMPILERS_KNOWN_TO_CMAKE33 AppleClang Clang GNU MSVC)

include(CheckCXXCompilerFlag)
if(COMB_CXX_STANDARD_FLAG MATCHES default)
  if("cxx_std_17" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    #TODO set BLT_CXX_STANDARD
    set(CMAKE_CXX_STANDARD 17)
  elseif("cxx_std_14" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    set(CMAKE_CXX_STANDARD 14)
  elseif("${CMAKE_CXX_COMPILER_ID}" IN_LIST COMPILERS_KNOWN_TO_CMAKE33)
    set(CMAKE_CXX_STANDARD 14)
  else() #cmake has no idea what to do, do it ourselves...
    foreach(flag_var "-std=c++17" "-std=c++1z" "-std=c++14" "-std=c++1y" "-std=c++11")
      CHECK_CXX_COMPILER_FLAG(${flag_var} COMPILER_SUPPORTS_${flag_var})
      if(COMPILER_SUPPORTS_${flag_var})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag_var}")
        break()
      endif()
    endforeach(flag_var)
  endif()
else(COMB_CXX_STANDARD_FLAG MATCHES default)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMB_CXX_STANDARD_FLAG}")
  message("Using C++ standard flag: ${COMB_CXX_STANDARD_FLAG}")
endif(COMB_CXX_STANDARD_FLAG MATCHES default)


set(CMAKE_CXX_EXTENSIONS OFF)

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
  if (ENABLE_COVERAGE)
    if(NOT ENABLE_CUDA)
      message(STATUS "Coverage analysis enabled")
      set(CMAKE_CXX_FLAGS "-coverage ${CMAKE_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "-coverage ${CMAKE_EXE_LINKER_FLAGS}")
    endif()
  endif()
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
  if ( NOT DEFINED COMB_NVCC_STD )
    set(COMB_NVCC_STD "c++11")
    # When we require cmake 3.8+, replace this with setting CUDA_STANDARD
    if(CUDA_VERSION_MAJOR GREATER "8")
      execute_process(COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc -std c++14 -ccbin ${CMAKE_CXX_COMPILER} .
                      ERROR_VARIABLE TEST_NVCC_ERR
                      OUTPUT_QUIET)
      if (NOT TEST_NVCC_ERR MATCHES "flag is not supported with the configured host compiler")
        set(COMB_NVCC_STD "c++14")
      endif()
    else()
    endif()
  endif()

  if (NOT COMB_HOST_CONFIG_LOADED)
    set(COMB_NVCC_FLAGS "-restrict -arch ${CUDA_ARCH} --expt-extended-lambda" CACHE STRING "")
    set(COMB_NVCC_FLAGS_RELEASE        "-O3"                                  CACHE STRING "")
    set(COMB_NVCC_FLAGS_RELWITHDEBINFO "-O2 -g -lineinfo"                     CACHE STRING "")
    set(COMB_NVCC_FLAGS_MINSIZEREL     "-Os"                                  CACHE STRING "")
    set(COMB_NVCC_FLAGS_DEBUG          "-O0 -g -G"                            CACHE STRING "")
  endif()

  if(COMB_ENABLE_COVERAGE)
    if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
      message(STATUS "Coverage analysis enabled")
      set(COMB_NVCC_FLAGS "${COMB_NVCC_FLAGS} -Xcompiler -coverage -Xlinker -coverage")
      set(CMAKE_EXE_LINKER_FLAGS "-coverage ${CMAKE_EXE_LINKER_FLAGS}")
    else()
      message(WARNING "Code coverage specified but not enabled -- GCC was not detected")
    endif()
  endif()

  if (NOT ENABLE_CLANG_CUDA)
    set(CMAKE_CUDA_FLAGS                ${COMB_NVCC_FLAGS}               )
    set(CMAKE_CUDA_FLAGS_RELEASE        ${COMB_NVCC_FLAGS_RELEASE}       )
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO ${COMB_NVCC_FLAGS_RELWITHDEBINFO})
    set(CMAKE_CUDA_FLAGS_MINSIZEREL     ${COMB_NVCC_FLAGS_MINSIZEREL}    )
    set(CMAKE_CUDA_FLAGS_DEBUG          ${COMB_NVCC_FLAGS_DEBUG}         )
  endif()
endif()
# end COMB_ENABLE_CUDA section
