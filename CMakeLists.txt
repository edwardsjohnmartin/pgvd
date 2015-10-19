cmake_minimum_required (VERSION 2.6)
project (PGVD)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#------------------------------------------------------------
# Use C++ 2011
#------------------------------------------------------------
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#------------------------------------------------------------
# Use OpenCL depending on setting
#------------------------------------------------------------
OPTION(OPENCL_ACCEL "Use OpenCL acceleration" OFF)
if(OPENCL_ACCEL)
  FIND_PACKAGE(OpenCL)
  if (OPENCL_FOUND)
    INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
    ADD_DEFINITIONS(-D__OPEN_CL_SUPPORT__)
  else (OPENCL_FOUND)
    set(OPENCL_LIBRARY "")
  endif (OPENCL_FOUND)
endif(OPENCL_ACCEL)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

#------------------------------------------------------------
# Source files
#------------------------------------------------------------
SET(SRCS
   ./viewer/gl_utils.cpp
   ./viewer/Octree2.cpp
   ./Karras.cpp
   ./OctreeUtils.cpp
   ./opencl/Geom.cpp
   ./bigint/BigInteger.cc
   ./bigint/BigUnsigned.cc
   ./bigint/BigIntegerAlgorithms.cc
   ./bigint/BigIntegerUtils.cc
   ./bigint/BigUnsignedInABase.cc
   ./options.cpp
#  ../opencl.cpp
#  ../octree.cpp
#  ../octree_gpu.cpp
#  ../gpu.cpp
#  ../ambiguous.cpp
#  ../vertex.cpp
#  ../vertex_network.cpp
#  ../mvertex_network.cpp
#  ../vertices_gpu_state.cpp
#  ../geometry_cpp.cpp
#  ../opencl/bit.cpp
#  ../opencl/geom.cpp
##  ../opencl/distance3.cpp
#  ../opencl/geometry.cpp
#  ../opencl/uvertex_network.cpp
#  ../opencl/cl_ambiguous.cpp
#  ../opencl/cl_octree.cpp
)

SET(PGVD2_SRCS
#  ../vector2.cpp
#  ./gl2d.cpp
#  ./gvdviewer2.cpp
#  ../jrs-triangle.c
)

SET(OCT2_SRCS
#  ../vector2.cpp
#  ./gl2d.cpp
#  ./octviewer2.cpp
#  ../jrs-triangle.c
#  ../karras.cpp
#  ../OctreeUtils.cpp
)

#SET(GVD3_SRCS
#  ../vector3.cpp
#  ../tile3.cpp
#  ./gl3d.cpp
#  ./gvdviewer3.cpp
#  ./texture.cpp
#  ./mesh.cpp
#  ./material.cpp
#  ./io.cpp
#  ../opencl/tribox.cpp
#  ../opencl/distance3.cpp
# ../orientation.cpp
#)

#SET(SEGMENT2_SRCS
#  ../vector2.cpp
#  ./gl2d.cpp
#  ./segment2.cpp
#  ../jrs-triangle.c
#)

ADD_DEFINITIONS(-DANSI_DECLARATORS -DTRILIBRARY)
ADD_DEFINITIONS(-U__STRICT_ANSI__)

FIND_PACKAGE(OpenGL)
FIND_PACKAGE(GLEW)
#FIND_PACKAGE(GLFW)
#FIND_PACKAGE(JPEG)
#FIND_PACKAGE(PNG)

#add_subdirectory(viewer/glfw-3.1.2)
#include_directories(viewer/glfw-3.1.2/include)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
#pkg_check_modules (GLIB2   glib-2.0)
include_directories(${GLFW_INCLUDE_DIRS})

IF(APPLE)
  ADD_DEFINITIONS(-D__MAC__ -DAPPLE)
  # WARNING: THE FOLLOWING LINE IS A HARD-CODED HACK TO GET GLFW TO LINK
  link_directories("/opt/local/lib")
ENDIF(APPLE)

option(BUILD_PGVD2 "Build 2D PGVD" ON)
if(BUILD_PGVD2)
  ADD_EXECUTABLE(pgvd2 ${SRCS} ${PGVD2_SRCS} viewer/main_pgvd2.cpp)
  set_target_properties (pgvd2 PROPERTIES COMPILE_DEFINITIONS "OCT2D")
  TARGET_LINK_LIBRARIES(pgvd2 ${GLFW_STATIC_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OPENCL_LIBRARY})
endif(BUILD_PGVD2)
