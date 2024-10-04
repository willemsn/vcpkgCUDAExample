CMAKE_MINIMUM_REQUIRED (VERSION 3.18)

PROJECT(VCPKGCuda VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
  
if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    # using GCC or CLANG

    # Both, GCC and Clang support the same option set
    # We first turn on Warnings
    # --> https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
    #
    # -Wall stands for All Warnings
    # -Wextra pushes that a bit further
    # -pedantic sets the compiler to use strict ISO C++
    # -Werror treats all warnings as errors so you have to fix them
    add_compile_options(-Wall -Wextra -pedantic) # -Werror)

    # Sanitizer options
    # The following will turn on Debugging (-g flag) and the Address Sanitizer
    # which is turned on with the -fsanitize=address option to the compiler.
    add_compile_options(-g -fsanitize=address,undefined)

    # When we turn on the sanitizer, you also need to instruct the linker that it
    # will be used which happens with the following link option command.
    add_link_options(-g -fsanitize=address,undefined)
    
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++
    
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    # MSVC uses a different warning flags
#    add_compile_options(/W4) # if we want warnings as errors, use this too -->  /WX

#    add_compile_options(/fsanitize=address /Zi /INCREMENTAL:NO)
#    add_link_options(/INCREMENTAL:NO /INFERASANLIBS)

    # Need these: https://github.com/MicrosoftDocs/cpp-docs/blob/main/docs/sanitizers/error-container-overflow.md#error-container-overflow
#    add_definitions(-D_DISABLE_VECTOR_ANNOTATION=1)
#    add_definitions(-D_DISABLE_STRING_ANNOTATION=1)
    
endif()
  
# ----------------------------------------------------------
# CUDA
# - check if CUDA is available
# ----------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)

   MESSAGE(STATUS "CUDA is available; CUDA_COMPILER=${CMAKE_CUDA_COMPILER}, Version=${CMAKE_CUDA_COMPILER_VERSION}")
   ENABLE_LANGUAGE( CUDA )

   if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
   endif()
   
else()
   MESSAGE(STATUS "CUDA is not available.")
endif()

# SET(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_52;-rdc=true;" )

#use fast math option is required for OptiX
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_61;-dlink;-rdc=true;--use_fast_math;--keep-device-functions")

# Attempt to find CUDA
IF(CMAKE_CUDA_COMPILER)
  MESSAGE(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
  SET(HAS_CUDA_SUPPORT ON CACHE BOOL "Determines if CUDA/GPU functionality is compiled into the code base" )

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  #INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/samples/common/inc)
  LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/lib64)
ENDIF()

find_package(CUDAToolkit REQUIRED)

add_executable(test_CUDARandomGen
  CUDARandomKernel.cu CUDARandomKernel.h
  test_CUDARandomGen.cpp
)

# set_target_properties(test_CUDARandomGen PROPERTIES
#                             CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(test_CUDARandomGen PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(test_CUDARandomGen PRIVATE ${CUDA_curand_LIBRARY})
