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
OPTION(OPENCL_ACCEL "Use OpenCL acceleration" ON)
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
# Header include directories
#------------------------------------------------------------
SET(INCL
    ./
    ./C
    ./opencl
    ./opencl/CLWrapper
    ./opencl/BufferCollection
    ./opencl/Kernels
)
INCLUDE_DIRECTORIES(${INCL})

#------------------------------------------------------------
# Source files
#------------------------------------------------------------
SET(SRCS
  ./Options.cpp

  ./viewer/gl_utils.cpp
  ./viewer/Octree2.cpp
  #./viewer/main_fit2.cpp
  #./viewer/main_pgvd2.cpp
  #./viewer/main_test2.cpp

  ./Karras.cpp

  ./OctreeUtils.cpp

  ./C/BuildOctree.c
  ./C/BigUnsigned.c
  ./C/BuildBRT.c
  ./C/z_order.c
  ./C/ParallelAlgorithms.c

  ./opencl/Geom.cpp
  ./opencl/CLWrapper/CLWrapper_Initializers.cpp
  ./opencl/CLWrapper/CLWrapper_Kernels.cpp
  ./opencl/Kernels/KernelBox.cpp
  #./opencl/BufferCollection/Buffer.cpp

  #These are new and may need to be moved...

  #headers
  ./BoundingBox.h
  ./CellIntersections.h
  ./Karras.h
  ./OctCell.h
  ./OctNode.h
  ./OctreeUtils.h
  ./Options.h
  ./Resln.h
  ./timer.h

  ./C/BigUnsigned.h
  ./C/bool.h
  ./C/BrtNode.h
  ./C/BuildBRT.h
  ./C/BuildOctree.h
  ./C/ParallelAlgorithms.h
  ./C/z_order.h

  ./viewer/Color.h
  ./viewer/gl_utils.h
  ./viewer/LinesProgram.h
  ./viewer/Polylines.h
  ./viewer/Octree2.h
  
  ./opencl/CLWrapper/CLWrapper.h
  ./opencl/Buffers/Buffer.h
  ./opencl/Buffers/Buffers.h
  ./opencl/Kernels/KernelBox.h
  ./opencl/vec_cpp.h
)

SET(2D_PGVD_SRCS
)

SET(OCT2_SRCS
)

#------------------------------------------------------------
# Maximum octree depth. Used for creating the Morton code
# integer.
#------------------------------------------------------------
SET(MAX_OCTREE_DEPTH 15)


ADD_DEFINITIONS(-DANSI_DECLARATORS -DTRILIBRARY)
ADD_DEFINITIONS(-U__STRICT_ANSI__)
ADD_DEFINITIONS(-DMAX_OCTREE_DEPTH=${MAX_OCTREE_DEPTH})


#NH find_path and find_libary mods for glew...
find_path(GLEW_INCLUDE_DIR GL/glew.h PATHS ${CMAKE_SOURCE_DIR}/dependencies/glew-1.13.0/include)
find_library(GLEW_LIBRARY NAMES GLEW glew32 glew glew32s PATHS ${CMAKE_SOURCE_DIR}/dependencies/glew-1.13.0/lib/Release/Win32 PATH_SUFFIXES lib64 Win32 Release)


FIND_PACKAGE(OpenGL)
FIND_PACKAGE(GLEW)
include_directories(${GLEW_INCLUDE_DIRS})


message(STATUS "GLEW_INCLUDE_DIR: ${GLEW_INCLUDE_DIRS}")
message(STATUS "GLEW_LIBRARY: ${GLEW_LIBRARY}")


IF(APPLE)
  ADD_DEFINITIONS(-D__MAC__ -DAPPLE)
ENDIF(APPLE)

option(BUILD_2D_PGVD "Build 2D PGVD" ON)
if(BUILD_2D_PGVD)
  ADD_EXECUTABLE(2D_PGVD ${SRCS} ${2D_PGVD_SRCS} viewer/main_pgvd2.cpp)
  set_target_properties (2D_PGVD PROPERTIES COMPILE_DEFINITIONS "OCT2D")
  TARGET_LINK_LIBRARIES (2D_PGVD glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OPENCL_LIBRARY})

  add_custom_command(TARGET 2D_PGVD PRE_BUILD
                   	COMMAND ${CMAKE_COMMAND} -E copy_directory
                       		${CMAKE_SOURCE_DIR}/C $<TARGET_FILE_DIR:2D_PGVD>/opencl/C)
  add_custom_command(TARGET 2D_PGVD PRE_BUILD
                   	COMMAND ${CMAKE_COMMAND} -E copy_directory
                       		${CMAKE_SOURCE_DIR}/opencl/Kernels/ $<TARGET_FILE_DIR:2D_PGVD>/opencl/Kernels)
  add_custom_command(TARGET 2D_PGVD PRE_BUILD
                   	COMMAND ${CMAKE_COMMAND} -E copy
                       		${CMAKE_SOURCE_DIR}/opencl/vec_cl.h $<TARGET_FILE_DIR:2D_PGVD>/opencl/vec_cl.h)
  add_custom_command(TARGET 2D_PGVD PRE_BUILD
                   	COMMAND ${CMAKE_COMMAND} -E copy
                       		${CMAKE_SOURCE_DIR}/opencl/dim.h $<TARGET_FILE_DIR:2D_PGVD>/opencl/dim.h)
  add_custom_command(TARGET 2D_PGVD PRE_BUILD
                   	COMMAND ${CMAKE_COMMAND} -E copy
                       		${CMAKE_SOURCE_DIR}/OctNode.h $<TARGET_FILE_DIR:2D_PGVD>/opencl/OctNode.h)
 add_custom_command(TARGET 2D_PGVD PRE_BUILD
                   	COMMAND ${CMAKE_COMMAND} -E copy_directory
                       		${CMAKE_SOURCE_DIR}/viewer/shaders $<TARGET_FILE_DIR:2D_PGVD>/opengl/shaders)
endif(BUILD_2D_PGVD)

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
