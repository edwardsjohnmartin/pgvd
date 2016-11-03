# Defining source files
set ( SRC
	"${CMAKE_CURRENT_LIST_DIR}/../include/cl.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/../include/clfw.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/clfw.cpp"
)

set ( INCLUDE_DIR
	"${OpenCL_INCLUDE_DIR}"
	"${CMAKE_CURRENT_LIST_DIR}/../include/"
)  

include_directories(${INCLUDE_DIR})
add_library(CLFW ${SRC} ${INCLUDE_DIR})