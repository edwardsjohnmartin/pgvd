# Source files that are 'shared' are copied to the build directory to be 'shared' with the GPU.
# Below are the files that are used both on the CPU and GPU. 
# Shaders and Kernels in this directory are simply copied over, and should not be added below. 

add_subdirectory(BigUnsigned)
add_subdirectory(BinaryRadixTree)
add_subdirectory(Boolean)
add_subdirectory(BoundingBox)
add_subdirectory(CellResolution)
add_subdirectory(Dimension)
add_subdirectory(Line)
add_subdirectory(Octree)
add_subdirectory(OctreeDefinitions)
add_subdirectory(OctreeResolution)
add_subdirectory(ParallelAlgorithms)
add_subdirectory(Quantize)
add_subdirectory(Vector)
add_subdirectory(ZOrder)

set(OCTREE_SHARED_SRC 
	${BIG_UNSIGNED_SRC}
	${BINARY_RADIX_TREE_SRC}
	${BOOLEAN_SRC}
	${BOUNDING_BOX_SRC}
	${CELL_RESOLUTION_SRC}
	${DIMENSION_SRC}
	${LINE_SRC}
	${OCTREE_C_SRC}
	${OCTREE_DEFINITIONS_SRC}
	${OCTREE_RESOLUTION_SRC}
	${PARALLEL_ALGORITHMS_SRC}
	${QUANTIZE_SRC}
	${VECTOR_SRC}
	${Z_ORDER_SRC}
	PARENT_SCOPE)
