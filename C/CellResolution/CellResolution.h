#pragma once
#ifndef __OPENCL_VERSION__
	#include "../Vector/vec_n.h"
  #include "../Octree/OctNode.h"
	#define __local
	#define __global
#else
	#include "./OpenCL/C/Vector/vec_n.h"
  #include "./OpenCL/C/Octree/OctNode.h"
#endif

//'i' is the indexes of the ambiguous lines
// If the two indexes don't match, 
typedef struct {
  int i[2]; 
} ConflictPair;

/* Ambiguous cells code */
unsigned char computeOutCode(float_n point, float_n min, float_n max);
void sub_v2v2(double_2 *a, double_2 b, double_2 c);
void dot_v2v2(double *dot, double_2 a, double_2 b);
void point_on_vn(double_2 *result, double_2 point, double_2 ray, double t);
bool v3_on_aasquare(float_3 point, float_3 min, float_3 normal, float width);
bool v2_on_aaedge(double_2 point, double_2 min, double_2 normal, float width);
bool doLineBoxTest(float_n *point1, float_n *point_2, float_n *minimum, float_n *maximum);

/* Run for each octnode in parallel */
int FindConflictCells(__global OctNode *octree, unsigned int octreeSize, float_n octreeCenter, float octreeWidth,
	__global ConflictPair* conflictPairs, __global int* smallestContainingCells, unsigned int numSCCS, __global Line* orderedLines, unsigned int numLines, __global float_2* points, unsigned int gid);

#ifndef __OPENCL_VERSION__
	#undef __local
	#undef __global
#endif
