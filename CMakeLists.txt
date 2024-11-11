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
    add_compile_options(-Wall -Wextra) # -pedantic) # -Werror)

    # Sanitizer options
    # The following will turn on Debugging (-g flag) and the Address Sanitizer
    # which is turned on with the -fsanitize=address option to the compiler.
#    add_compile_options(-g -fsanitize=address,undefined)

    # When we turn on the sanitizer, you also need to instruct the linker that it
    # will be used which happens with the following link option command.
#    add_link_options(-g -fsanitize=address,undefined)
    
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
INCLUDE(CheckLanguage)

FIND_PACKAGE(CUDAToolkit REQUIRED)
CHECK_LANGUAGE(CUDA)

if (CMAKE_CUDA_COMPILER)

  MESSAGE(STATUS "CUDA Toolkit Root: ${CUDAToolkit_ROOT}")
  MESSAGE(STATUS "CUDA Toolkit Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")
  MESSAGE(STATUS "CUDA Toolkit Libraries: ${CUDAToolkit_LIBRARIES}")
  MESSAGE(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
  MESSAGE(STATUS "CUDA Toolkit Binary Directory: ${CUDAToolkit_BIN_DIR}")

  MESSAGE(STATUS "CUDA is available; CUDA_COMPILER=${CMAKE_CUDA_COMPILER}, Version=${CMAKE_CUDA_COMPILER_VERSION}")
  MESSAGE(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
  ENABLE_LANGUAGE( CUDA )

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
   
else()
   MESSAGE(STATUS "CUDA is not available.")
endif()

# ----------------------------------------------------------
# NetCDF
# ----------------------------------------------------------
SET(NETCDF_CXX "YES")
FIND_PACKAGE(netCDFCxx CONFIG REQUIRED)
IF(netCDFCxx_FOUND)
  MESSAGE(STATUS "Found NetCDF CXX: ${NETCDF_C_INCLUDE_DIR} ${NETCDF_C_LIBRARY}")
  INCLUDE_DIRECTORIES(${NETCDF_C_INCLUDE_DIR})
ENDIF(netCDFCxx_FOUND)

#  target_link_libraries(main PRIVATE netCDF::netcdf netCDF::netcdf-cxx4)

FIND_PACKAGE(netCDF CONFIG REQUIRED)
IF(netCDF_FOUND)
  MESSAGE(STATUS "Found NetCDF - Version: ${NetCDFVersion}, ${netCDF_INCLUDE_DIR}, ${netCDF_LIB_DIR}, ${netCDF_LIBRARIES}, ${netCDF_LDFLAGS}, ${netCDF_C_CPP_FLAGS}")
#  ADD_LINK_OPTIONS("${netCDF_LDFLAGS}")
  INCLUDE_DIRECTORIES(${netCDF_INCLUDE_DIR})
  LINK_DIRECTORIES(${netCDF_LIB_DIR})
ENDIF(netCDF_FOUND)


add_executable(CUDARandomGen
  CUDARandomKernel.cu CUDARandomKernel.h
  CUDARandomGen.cpp
)

target_include_directories(CUDARandomGen PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(CUDARandomGen PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(CUDARandomGen PRIVATE ${CUDA_curand_LIBRARY})



add_library(CUDALib STATIC
  CUDA_DeviceFunctions.cu CUDA_DeviceFunctions.h
)
target_include_directories(CUDALib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(CUDALib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(CUDALib PROPERTIES CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-rdc=true" )  # shouldn't be needed with sep comp above

add_executable(CUDALibraryTest
  CUDA_LibraryTest.cu CUDA_DeviceFunctions.h
)
target_include_directories(CUDALibraryTest PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(CUDALibraryTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CUDALibraryTest PRIVATE CUDALib)
target_link_libraries(CUDALibraryTest PRIVATE ${CUDA_LIBRARIES})



Include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.0.1 # or a later release
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_CUDARandomGen
  CUDARandomKernel.cu CUDARandomKernel.h
  test_CUDARandomGen.cpp)
target_include_directories(test_CUDARandomGen PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_CUDARandomGen ${CUDA_LIBRARIES})
target_link_libraries(test_CUDARandomGen ${CUDA_curand_LIBRARY})
target_link_libraries(test_CUDARandomGen Catch2::Catch2WithMain)
