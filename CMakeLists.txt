cmake_minimum_required (VERSION 2.6)

project (PGVD)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

#------------------------------------------------------------
# Use C++ 2011
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
include_directories(${OPENGL_INCLUDE_DIR})

# use GLFW
find_package(GLFW REQUIRED)
include_directories(${GLFW_INCLUDE_DIR})

# use GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})

# use GLM
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIR})

# use OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})

# use CLFW (OpenCL must be included first.)
add_subdirectory(Dependencies/CLFW)
include_directories(${CLFW_INCLUDE_DIR})

#------------------------------------------------------------
# Include directories
#------------------------------------------------------------
add_subdirectory(C)
add_subdirectory(CPP)

# #------------------------------------------------------------
# # Maximum octree depth. Used for creating the Morton code
# # integer.
# #------------------------------------------------------------
# SET(MAX_OCTREE_DEPTH 15)


# ADD_DEFINITIONS(-DANSI_DECLARATORS -DTRILIBRARY)
# ADD_DEFINITIONS(-U__STRICT_ANSI__)
# ADD_DEFINITIONS(-DMAX_OCTREE_DEPTH=${MAX_OCTREE_DEPTH})


# #NH find_path and find_libary mods for glew...
# find_path(GLEW_INCLUDE_DIR GL/glew.h PATHS ${CMAKE_SOURCE_DIR}/dependencies/glew-1.13.0/include)
# find_library(GLEW_LIBRARY NAMES GLEW glew32 glew glew32s PATHS ${CMAKE_SOURCE_DIR}/dependencies/glew-1.13.0/lib/Release/Win32 PATH_SUFFIXES lib64 Win32 Release)



# include_directories(${GLEW_INCLUDE_DIR})

# IF(APPLE)
#   ADD_DEFINITIONS(-D__MAC__ -DAPPLE)
# ENDIF(APPLE)

option(BUILD_2D_PGVD "Build 2D PGVD" ON)
if(BUILD_2D_PGVD)

  #2D GVDS require quadtrees.
  add_executable (QUADTREE ${OCTREE_CPP_SRC} ${OCTREE_C_SRC})
  set_target_properties (QUADTREE PROPERTIES COMPILE_DEFINITIONS "QUADTREE")
  target_link_libraries (QUADTREE ${OPENCL_LIBRARY} ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${GLEW_LIBRARY} CLFW )

	add_custom_target(
	    COPY_KERNELS
	    COMMAND ${CMAKE_COMMAND} -E copy_directory 
	    ${CMAKE_SOURCE_DIR}/OpenCL 
	    ${CMAKE_CURRENT_BINARY_DIR}/OpenCL

	    COMMAND ${CMAKE_COMMAND} -E copy_directory 
	    ${CMAKE_SOURCE_DIR}/C 
	    ${CMAKE_CURRENT_BINARY_DIR}/OpenCL/C
	)

	add_custom_target(
	    COPY_SHADERS
	    COMMAND ${CMAKE_COMMAND} -E copy_directory 
	    ${CMAKE_SOURCE_DIR}/OpenGL 
	    ${CMAKE_CURRENT_BINARY_DIR}/OpenGL
	)
	add_dependencies(QUADTREE COPY_KERNELS COPY_SHADERS)

#   add_custom_command(TARGET 2D_PGVD PRE_BUILD
#                    	COMMAND ${CMAKE_COMMAND} -E copy
#                        		${CMAKE_SOURCE_DIR}/opencl/opencl_sources.txt $<TARGET_FILE_DIR:2D_PGVD>/opencl_sources.txt)
#   add_custom_command(TARGET 2D_PGVD PRE_BUILD
#                    	COMMAND ${CMAKE_COMMAND} -E copy_directory
#                        		${CMAKE_SOURCE_DIR}/C $<TARGET_FILE_DIR:2D_PGVD>/opencl/C)
#   add_custom_command(TARGET 2D_PGVD PRE_BUILD
#                    	COMMAND ${CMAKE_COMMAND} -E copy_directory
#                        		${CMAKE_SOURCE_DIR}/opencl/Kernels/ $<TARGET_FILE_DIR:2D_PGVD>/opencl/Kernels)
#   add_custom_command(TARGET 2D_PGVD PRE_BUILD
#                    	COMMAND ${CMAKE_COMMAND} -E copy_directory
#                        		${CMAKE_SOURCE_DIR}/viewer/shaders $<TARGET_FILE_DIR:2D_PGVD>/opengl/shaders)
#   add_custom_command(TARGET 2D_PGVD PRE_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_directory
#                           ${CMAKE_SOURCE_DIR}/opengl/shaders $<TARGET_FILE_DIR:2D_PGVD>/opengl/shaders)
endif(BUILD_2D_PGVD)

# option(BUILD_2D_PGVD_UNIT_TESTS "2D PGVD Unit Tests" ON)
# if(BUILD_2D_PGVD_UNIT_TESTS)
# #Adding files to target
#   ADD_EXECUTABLE(2D_PGVD_UNIT_TESTS ${UNIT_TEST_SOURCES} tests/main.cpp)
#   set_target_properties (2D_PGVD_UNIT_TESTS PROPERTIES COMPILE_DEFINITIONS "OCT2D")
#   TARGET_LINK_LIBRARIES (2D_PGVD_UNIT_TESTS glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${OPENCL_LIBRARY} CLFW)

# #Custom Build Commands
#   add_custom_command(TARGET 2D_PGVD_UNIT_TESTS PRE_BUILD
#                    	COMMAND ${CMAKE_COMMAND} -E copy
#                        		${CMAKE_SOURCE_DIR}/opencl/opencl_sources.txt $<TARGET_FILE_DIR:2D_PGVD_UNIT_TESTS>/opencl_sources.txt)
#   add_custom_command(TARGET 2D_PGVD_UNIT_TESTS PRE_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_directory
#                           ${CMAKE_SOURCE_DIR}/C $<TARGET_FILE_DIR:2D_PGVD_UNIT_TESTS>/opencl/C)
#   add_custom_command(TARGET 2D_PGVD_UNIT_TESTS PRE_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_directory
#                           ${CMAKE_SOURCE_DIR}/opencl/Kernels/ $<TARGET_FILE_DIR:2D_PGVD_UNIT_TESTS>/opencl/Kernels)
#   add_custom_command(TARGET 2D_PGVD_UNIT_TESTS PRE_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_directory
#                           ${CMAKE_SOURCE_DIR}/viewer/shaders $<TARGET_FILE_DIR:2D_PGVD_UNIT_TESTS>/opengl/shaders)
#   add_custom_command(TARGET 2D_PGVD PRE_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy_directory
#                         ${CMAKE_SOURCE_DIR}/opengl/shaders $<TARGET_FILE_DIR:2D_PGVD>/opengl/shaders)
# endif(BUILD_2D_PGVD_UNIT_TESTS)

# option(BUILD_TEST2 "Build 2D TEST" OFF)
# if(BUILD_TEST2)
#   ADD_EXECUTABLE(test2 ${SRCS} viewer/main_test2.cpp)
#   set_target_properties (test2 PROPERTIES COMPILE_DEFINITIONS "OCT2D")
#   TARGET_LINK_LIBRARIES(test2 glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${OPENCL_LIBRARY})
# endif(BUILD_TEST2)

# option(BUILD_FIT2 "Build 2D FIT" OFF)
# if(BUILD_FIT2)
#   ADD_EXECUTABLE(fit2 ${SRCS} viewer/main_fit2.cpp)
#   set_target_properties (fit2 PROPERTIES COMPILE_DEFINITIONS "OCT2D")
#   TARGET_LINK_LIBRARIES(fit2 glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${OPENCL_LIBRARY})
# endif(BUILD_FIT2)
