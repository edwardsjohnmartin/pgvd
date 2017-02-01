cmake_minimum_required (VERSION 2.6)
include(${CMAKE_CURRENT_SOURCE_DIR}/CopyIfDifferent.txt)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Must go before the project command
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.10" CACHE STRING "Minimum OS X deployment version")

project (PGVD)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(SRC_PATH "Sources")

#------------------------------------------------------------
# Use C++ 11
#------------------------------------------------------------

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#------------------------------------------------------------
# Use external dependencies
#------------------------------------------------------------


find_package(OpenGL REQUIRED)
include_directories(SYSTEM ${OPENGL_INCLUDE_DIR})

# use GLFW
find_package(GLFW REQUIRED)
include_directories(SYSTEM ${GLFW_INCLUDE_DIR})

# use GLEW
find_package(GLEW REQUIRED)
include_directories(SYSTEM ${GLEW_INCLUDE_DIR})

# use GLM
find_package(GLM REQUIRED)
include_directories(SYSTEM ${GLM_INCLUDE_DIR})

# use OpenCL
find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OPENCL_INCLUDE_DIR})


# use CLFW (OpenCL must be included first.)
add_subdirectory(Dependencies/CLFW)
include_directories(${CLFW_INCLUDE_DIR})

#------------------------------------------------------------
# Source Files
#------------------------------------------------------------
add_subdirectory(Sources)

# For IDE folder hierarchy generation
file(
    GLOB_RECURSE SRC_LIST 
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${SRC_PATH}/*.c*"
    "${SRC_PATH}/*.h*"
    "${SRC_PATH}/*.txt*"
    "${SRC_PATH}/*.frag*"
    "${SRC_PATH}/*.vert*"
)

if(APPLE)
  add_definitions(-D__MAC__ -DAPPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif(APPLE)

#2D GVDS require quadtrees.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Sources)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Dependencies)

# UNIT TEST TARGET
add_executable (Unit_Tests "${OCTREE_TEST}")
set_target_properties (Unit_Tests PROPERTIES COMPILE_DEFINITIONS "QUADTREE;OCT2D;__OPEN_CL_SUPPORT__")
target_link_libraries (Unit_Tests ${OPENCL_LIBRARY} ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${GLEW_LIBRARY} CLFW)

# QUADTREE TARGET
add_executable (Quadtree "${OCTREE_SRC}")
set_target_properties (Quadtree PROPERTIES COMPILE_DEFINITIONS "QUADTREE;OCT2D;__OPEN_CL_SUPPORT__")
target_link_libraries (Quadtree ${OPENCL_LIBRARY} ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${GLEW_LIBRARY} CLFW)

# For IDE folder hierarchy generation
foreach(_source IN ITEMS ${SRC_LIST})
	get_filename_component(_source_path "${_source}" PATH)
	string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
	source_group("${_source_path_msvc}" FILES "${_source}")
endforeach()  

	# Temporary target to copy the kernel sources to where the main target is built.
copy_touched (
 ${CMAKE_SOURCE_DIR}/Sources
 ${CMAKE_CURRENT_BINARY_DIR}/Sources
 CopySourcesToBinary
)

copy_touched (
 ${CMAKE_SOURCE_DIR}/TestData
 ${CMAKE_CURRENT_BINARY_DIR}/TestData
 CopyUnitTestDataToBinary
)

add_dependencies(Unit_Tests CopyUnitTestDataToBinary)
add_dependencies(Unit_Tests CopySourcesToBinary)
add_dependencies(Quadtree CopySourcesToBinary)

# Most platforms don't support SPIR-V yet. Eventually we'll switch to this though.
#string(REPLACE ";" " " _KERNEL_SRC "${KERNEL_SRC}")
#separate_arguments(KERNEL_SRC WINDOWS_COMMAND "${_KERNEL_SRC}")

#add_custom_target(
#	Compile_Kernels
#	COMMAND clang -cc1 -emit-spirv -triple=spir-unknown-unknown -cl-std=CL2.0 -I ./Sources -D OpenCL -include ${KERNEL_SRC} -o kernel.spir
#)
