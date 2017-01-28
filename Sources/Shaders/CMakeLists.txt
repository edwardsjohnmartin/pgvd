add_subdirectory(Programs)
set (Shaders_SRC 
	${Programs_SRC}
	${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
	${CMAKE_CURRENT_SOURCE_DIR}/Shaders.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Shaders.hpp
	PARENT_SCOPE)