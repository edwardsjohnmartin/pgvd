cmake_minimum_required (VERSION 2.6)
project (PGVD)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#------------------------------------------------------------
# build glfw
#------------------------------------------------------------
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_USE_CHDIR "Make glfwInit chdir to Contents/Resources" OFF)
option(GLFW_USE_MENUBAR "Populate the menu bar on first window creation" OFF)
add_subdirectory (glfw)
include_directories(glfw/include)

#------------------------------------------------------------
# Use C++ 2011
#------------------------------------------------------------
if(WIN32)
  message(STATUS "Please verify you are using a compiler with C++11 support.")
else()
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
   ./Options.cpp
)

SET(PGVD2_SRCS
)

SET(OCT2_SRCS
)

ADD_DEFINITIONS(-DANSI_DECLARATORS -DTRILIBRARY)
ADD_DEFINITIONS(-U__STRICT_ANSI__)

FIND_PACKAGE(OpenGL)
FIND_PACKAGE(GLEW)
include_directories(${GLEW_INCLUDE_DIRS})

IF(APPLE)
  ADD_DEFINITIONS(-D__MAC__ -DAPPLE)
ENDIF(APPLE)

option(BUILD_PGVD2 "Build 2D PGVD" ON)
if(BUILD_PGVD2)
  ADD_EXECUTABLE(pgvd2 ${SRCS} ${PGVD2_SRCS} viewer/main_pgvd2.cpp)
  set_target_properties (pgvd2 PROPERTIES COMPILE_DEFINITIONS "OCT2D")
  TARGET_LINK_LIBRARIES(pgvd2 glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OPENCL_LIBRARY})
endif(BUILD_PGVD2)

option(BUILD_TEST2 "Build 2D TEST" OFF)
if(BUILD_TEST2)
  ADD_EXECUTABLE(test2 ${SRCS} viewer/main_test2.cpp)
  set_target_properties (test2 PROPERTIES COMPILE_DEFINITIONS "OCT2D")
  TARGET_LINK_LIBRARIES(test2 glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OPENCL_LIBRARY})
endif(BUILD_TEST2)

option(BUILD_FIT2 "Build 2D FIT" OFF)
if(BUILD_FIT2)
  ADD_EXECUTABLE(fit2 ${SRCS} viewer/main_fit2.cpp)
  set_target_properties (fit2 PROPERTIES COMPILE_DEFINITIONS "OCT2D")
  TARGET_LINK_LIBRARIES(fit2 glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OPENCL_LIBRARY})
endif(BUILD_FIT2)
