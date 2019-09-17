cmake_minimum_required(VERSION 3.0.2)
project(testbench)

# Options
option(USE_CUDA_MEMCPY "Use the normal CUDA memory model with CUDA memcopy operations to move to and from the device." ON)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -Wno-long-long -fno-omit-frame-pointer")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(default_build_type "Release")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Find CUDA
find_package(CUDA REQUIRED)
include_directories ( ${CUDA_INCLUDE_DIRS} )
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
if(NOT COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
	include(${CMAKE_SOURCE_DIR}/cmake/FindCUDA/select_compute_arch.cmake)
endif()
find_cuda_helper_libs(cufftw)
find_cuda_helper_libs(nvToolsExt)

# Set NVCC options
if(NOT COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
  include(${CMAKE_SOURCE_DIR}/cmake/FindCUDA/select_compute_arch.cmake)
endif()
set(CUDA_ARCH_LIST "Auto" CACHE STRING "CUDA GPU architecture for building the code")
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${CUDA_ARCH_LIST})
list( APPEND CUDA_NVCC_FLAGS -O3 -std=c++11 ${ARCH_FLAGS} --default-stream per-thread) # --gpu-architecture sm_62 )

# Add custom compile options
add_definitions(-DCUFFT)
IF(USE_CUDA_MEMCPY)
  add_definitions(-DUSE_CUDA_MEMCPY )
  MESSAGE(STATUS "CUDA using memcpy operations")
ENDIF()

# Create compilation target
include_directories( ${CMAKE_BINARY_DIR}/ ${CMAKE_SOURCE_DIR}/kcf/ ${CMAKE_SOURCE_DIR})
cuda_add_executable( testbench main.cpp testbench.cpp testbench.h utility_host.hpp utility_func.cu utility_func.cuh kcf/cuda_error_check.hpp kcf/complexmat.cu kcf/complexmat.cuh )
target_link_libraries( testbench ${CUDA_cufft_LIBRARY} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
