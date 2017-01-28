# -----------------------------------------------------
# Add every subdirectory
# -----------------------------------------------------
MACRO(GETSUBDIRLIST RESULT CURDIR)
  FILE(GLOB CHILDREN RELATIVE ${CURDIR} ${CURDIR}/*)
  SET(DIRLIST "")
  FOREACH(CHILD ${CHILDREN})
    IF(IS_DIRECTORY ${CURDIR}/${CHILD})
      LIST(APPEND DIRLIST ${CHILD})
    ENDIF()
  ENDFOREACH()
  SET(${RESULT} ${DIRLIST})
ENDMACRO()

GETSUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})

FOREACH(SUBDIR ${SUBDIRS})
  ADD_SUBDIRECTORY(${SUBDIR})
ENDFOREACH()

MACRO(GETFILES RESULT SUFFIX)
LIST(APPEND _${SUFFIX} "")
FOREACH(SUBDIR ${SUBDIRS})
	LIST(APPEND _${SUFFIX} ${${SUBDIR}_${SUFFIX}})
ENDFOREACH()
SET(${RESULT} ${_${SUFFIX}})
ENDMACRO()


# -----------------------------------------------------
# Standard project source files
# -----------------------------------------------------
GETFILES(_OCTREE_SRC SRC)

set(OCTREE_SRC ${_OCTREE_SRC} 
  ${CMAKE_CURRENT_SOURCE_DIR}/main_pgvd2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
  PARENT_SCOPE)

GETFILES(_OCTREE_TEST TEST)
set(OCTREE_TEST ${_OCTREE_SRC} 
	${_OCTREE_TEST} 
  ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
  PARENT_SCOPE)

# -----------------------------------------------------
# Shader files
# -----------------------------------------------------
#GETFILES(thingy KRNL)
#MESSAGE(STATUS "${thingy}")

# -----------------------------------------------------
# Kernel sources
# -----------------------------------------------------
GETFILES(KERNEL_FILES KRNL)
GETFILES(_KERNEL_SRC KRNLSRC)
SET(_KERNEL_SRC "opencl.h;${KERNEL_FILES};${_KERNEL_SRC}")
set(KERNEL_SRC ${_KERNEL_SRC} 
	PARENT_SCOPE)
