add_subdirectory(Programs)
set (SHADERS_SRC 
	${PROGRAMS_SRC}
	${CMAKE_CURRENT_SOURCE_DIR}/Shaders.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Shaders.hpp
	PARENT_SCOPE)