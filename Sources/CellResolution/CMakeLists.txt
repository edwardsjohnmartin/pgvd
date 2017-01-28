set (CellResolution_SRC 
	${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
	${CMAKE_CURRENT_SOURCE_DIR}/ConflictCellDetection.h
	${CMAKE_CURRENT_SOURCE_DIR}/ConflictCellDetection.cpp	
	${CMAKE_CURRENT_SOURCE_DIR}/LineTransform.h
	${CMAKE_CURRENT_SOURCE_DIR}/Conflict.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Conflict.h
	PARENT_SCOPE)

 set (CellResolution_KRNLSRC 
 	${CMAKE_CURRENT_SOURCE_DIR}/ConflictCellDetection.cpp	
 	${CMAKE_CURRENT_SOURCE_DIR}/Conflict.cpp
 	PARENT_SCOPE)