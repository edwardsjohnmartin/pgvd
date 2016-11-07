cmake_minimum_required (VERSION 2.6)

# Must go before the project command
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.10" CACHE STRING "Minimum OS X deployment version")

project (PGVD)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

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
# Include directories
#------------------------------------------------------------
add_subdirectory(SharedSources)
add_subdirectory(Sources)

if(APPLE)
  add_definitions(-D__MAC__ -DAPPLE)
endif(APPLE)

option(BUILD_2D_PGVD "Build 2D PGVD" ON)
if(BUILD_2D_PGVD)
	#2D GVDS require quadtrees.
	add_executable (QUADTREE ${OCTREE_SHARED_SRC} ${OCTREE_SRC})
	set_target_properties (QUADTREE PROPERTIES COMPILE_DEFINITIONS "QUADTREE;OCT2D;__OPEN_CL_SUPPORT__")
	target_link_libraries (QUADTREE ${OPENCL_LIBRARY} ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${GLEW_LIBRARY} CLFW )

  	add_custom_target(
  		COPY_SHARED_SRC
  		COMMAND ${CMAKE_COMMAND} -E copy_directory 
	    ${CMAKE_SOURCE_DIR}/SharedSources 
	    ${CMAKE_CURRENT_BINARY_DIR}/SharedSources
  	)

  	# Copy the shared sources whenever the main target is built.
	add_dependencies(QUADTREE COPY_SHARED_SRC)
endif(BUILD_2D_PGVD)

