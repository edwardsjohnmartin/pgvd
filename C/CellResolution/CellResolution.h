#pragma once
#ifndef __OPENCL_VERSION__
	#include "../Vector/vec_n.h"
  #include "../Octree/OctNode.h"
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
unsigned char computeOutCode(floatn point, floatn min, floatn max);
void sub_v2v2(cl_double2 *a, cl_double2 b, cl_double2 c);
void dot_v2v2(double *dot, cl_double2 a, cl_double2 b);
void point_on_vn(cl_double2 *result, cl_double2 point, cl_double2 ray, double t);
bool v3_on_aasquare(cl_float3 point, cl_float3 min, cl_float3 normal, float width);
bool v2_on_aaedge(cl_double2 point, cl_double2 min, cl_double2 normal, float width);
bool doLineBoxTest(const floatn *point1, const floatn *point2, const floatn *minimum, const floatn *maximum);

/* Run for each octnode in parallel */
cl_int FindConflictCells(OctNode *octree, unsigned int octreeSize, floatn octreeCenter, float octreeWidth,
  ConflictPair* conflictPairs, int* smallestContainingCells, unsigned int numSCCS, Line* orderedLines, unsigned int numLines, float2* points, unsigned int gid);