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

if (ENABLE_MPI)
  if(MPI_FOUND)
    message(STATUS "MPI Enabled")
  else()
    message(FATAL_ERROR "MPI NOT FOUND")
  endif()
endif()


if (ENABLE_OPENMP)
  if(OPENMP_FOUND)
    message(STATUS "OpenMP Enabled")
  else()
    message(FATAL_ERROR "OpenMP NOT FOUND")
  endif()
endif()


if (ENABLE_CUDA)
  if(CUDA_FOUND)
    message(STATUS "Cuda Enabled")
  else()
    message(FATAL_ERROR "Cuda NOT FOUND")
  endif()
endif()


if (ENABLE_GDSYNC)
  message(STATUS "FindGDSYNC.cmake ${PROJECT_SOURCE_DIR}/cmake")
  set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
  find_package(gdsync REQUIRED)

  if (GDSYNC_FOUND)
    message(STATUS "GDSYNC Enabled")
    message(STATUS "GDSYNC  Compile Flags:  ${GDSYNC_CXX_COMPILE_FLAGS}")
    message(STATUS "GDSYNC  Include Path:   ${GDSYNC_INCLUDE_PATH}")
    message(STATUS "GDSYNC  Link Flags:     ${GDSYNC_CXX_LINK_FLAGS}")
    message(STATUS "GDSYNC  Libraries:      ${GDSYNC_CXX_LIBRARIES}")
    message(STATUS "GDSYNC  Device Arch:    ${GDSYNC_ARCH}")
  else()
    message(FATAL_ERROR "gdsync NOT FOUND")
  endif()

  # register GDSYNC with blt
  blt_register_library(NAME gdsync
                       INCLUDES ${GDSYNC_CXX_INCLUDE_PATH}
                       LIBRARIES ${GDSYNC_CXX_LIBRARIES}
                       COMPILE_FLAGS ${GDSYNC_CXX_COMPILE_FLAGS}
                       LINK_FLAGS    ${GDSYNC_CXX_LINK_FLAGS}
                       DEFINES USE_GDSYNC)
endif()


if (ENABLE_GPUMP)
  message(STATUS "FindGPUMP.cmake ${PROJECT_SOURCE_DIR}/cmake")
  set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
  find_package(gpump REQUIRED)

  if (GPUMP_FOUND)
    message(STATUS "GPUMP Enabled")
    message(STATUS "GPUMP  Compile Flags:  ${GPUMP_CXX_COMPILE_FLAGS}")
    message(STATUS "GPUMP  Include Path:   ${GPUMP_INCLUDE_PATH}")
    message(STATUS "GPUMP  Link Flags:     ${GPUMP_CXX_LINK_FLAGS}")
    message(STATUS "GPUMP  Libraries:      ${GPUMP_CXX_LIBRARIES}")
    message(STATUS "GPUMP  Device Arch:    ${GPUMP_ARCH}")
  else()
    message(FATAL_ERROR "gpump NOT FOUND")
  endif()

  # register GPUMP with blt
  blt_register_library(NAME gpump
                       INCLUDES ${GPUMP_CXX_INCLUDE_PATH}
                       LIBRARIES ${GPUMP_CXX_LIBRARIES}
                       COMPILE_FLAGS ${GPUMP_CXX_COMPILE_FLAGS}
                       LINK_FLAGS    ${GPUMP_CXX_LINK_FLAGS}
                       DEFINES USE_GPUMP)
endif()


if (ENABLE_MP)
  message(STATUS "FindMP.cmake ${PROJECT_SOURCE_DIR}/cmake")
  set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
  find_package(MP REQUIRED)

  if (MP_FOUND)
    message(STATUS "MP Enabled")
    message(STATUS "MP  Compile Flags:  ${MP_CXX_COMPILE_FLAGS}")
    message(STATUS "MP  Include Path:   ${MP_INCLUDE_PATH}")
    message(STATUS "MP  Link Flags:     ${MP_CXX_LINK_FLAGS}")
    message(STATUS "MP  Libraries:      ${MP_CXX_LIBRARIES}")
    message(STATUS "MP  Device Arch:    ${MP_ARCH}")
  else()
    message(FATAL_ERROR "mp NOT FOUND")
  endif()

  # register MP with blt
  blt_register_library(NAME mp
                       INCLUDES ${MP_CXX_INCLUDE_PATH}
                       LIBRARIES ${MP_CXX_LIBRARIES}
                       COMPILE_FLAGS ${MP_CXX_COMPILE_FLAGS}
                       LINK_FLAGS    ${MP_CXX_LINK_FLAGS}
                       DEFINES USE_MP)
endif()


if (ENABLE_UMR)
  message(STATUS "FindUMR.cmake ${PROJECT_SOURCE_DIR}/cmake")
  set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
  find_package(UMR REQUIRED)

  if (UMR_FOUND)
    message(STATUS "UMR Enabled")
    message(STATUS "UMR  Compile Flags:  ${UMR_CXX_COMPILE_FLAGS}")
    message(STATUS "UMR  Include Path:   ${UMR_INCLUDE_PATH}")
    message(STATUS "UMR  Link Flags:     ${UMR_CXX_LINK_FLAGS}")
    message(STATUS "UMR  Libraries:      ${UMR_CXX_LIBRARIES}")
    message(STATUS "UMR  Device Arch:    ${UMR_ARCH}")
  else()
    message(FATAL_ERROR "UMR NOT FOUND")
  endif()

  # register UMR with blt
  blt_register_library(NAME UMR
                       INCLUDES ${UMR_CXX_INCLUDE_PATH}
                       LIBRARIES ${UMR_CXX_LIBRARIES}
                       COMPILE_FLAGS ${UMR_CXX_COMPILE_FLAGS}
                       LINK_FLAGS    ${UMR_CXX_LINK_FLAGS}
                       DEFINES USE_UMR)
endif()

if (ENABLE_RAJA)
  set(RAJA_ENABLE_EXERCISES ${ENABLE_EXERCISES} CACHE BOOL "")
  if (DEFINED RAJA_DIR)
    find_package(RAJA REQUIRED)

    if (RAJA_FOUND)
    else()
      message(FATAL_ERROR "RAJA NOT FOUND")
    endif()

    blt_print_target_properties(TARGET RAJA)
  else ()
    add_subdirectory(tpl/RAJA)
  endif ()
endif ()
