#pragma once
#define OCT2D
#include "OctreeDefinitions/defs.h"
#include "Vector/vec.h"
#include "Octree/OctNode.h"
#include "Line/Line.h"

#ifndef OpenCL
#define __local
#define __global
#endif

//'i' is the indexes of the ambiguous lines
// If the two indexes don't match, 
typedef struct {
	cl_int color;
	cl_int q1[2];
	cl_int q2[2];
	cl_int width;
	intn origin;
} Conflict;

#ifdef __cplusplus
inline std::ostream& operator<<(std::ostream& os, const Conflict& c) {
	os << "color: " << c.color << " i:[" << c.q1[0] << ", " << c.q1[1] << "]" << " width: " << c.width << " origin: " << c.origin;
	return os;
}
#endif

/* Ambiguous cells code */
bool liangBarskey(floatn *min, floatn *max, floatn *p1, floatn *p2, int gid, int debug);

#define dot(u,v)   ((u).x * (v).x + (u).y * (v).y)

//Adapted from Dan Sunday
//Copyright 2001 softSurfer, 2012
//Segment = P1 - P0
inline float distPointToSegment(intn P, intn P0, intn P1)
{
	intn v = P1 - P0;
	intn w = P - P0;

	float c1 = dot(w, v);
	if (c1 <= 0)
	return sqrt((float)dot(P - P0, P - P0));

	float c2 = dot(v, v);
	if (c2 <= c1)
	return sqrt((float)dot(P - P1, P - P1));

	float b = c1 / c2;
	floatn Pb = convert_floatn(P0) + (convert_floatn(v) * b);
	return sqrt((float)dot(convert_floatn(P) - Pb, convert_floatn(P) - Pb));
}

inline bool SameSign(float a, float b) {
	return a*b >= 0.0f;
}

inline float distSegmentToSegment(intn A0, intn A1, intn B0, intn B1) {
	float d1 = ((B0.x - A0.x)*(A1.y - A0.y)) - ((B0.y - A0.y)*(A1.x - A0.x));
	float d2 = ((B1.x - A0.x)*(A1.y - A0.y)) - ((B1.y - A0.y)*(A1.x - A0.x));

	if (!SameSign(d1, d2)) return 0;

	float A0B0B1 = distPointToSegment(A0, B0, B1);
	float A1B0B1 = distPointToSegment(A1, B0, B1);
	float B0A0A1 = distPointToSegment(B0, A0, A1);
	float B1A0A1 = distPointToSegment(B1, A0, A1);
	float min = A0B0B1;
	min = (A1B0B1 < min) ? A1B0B1 : min;
	min = (B0A0A1 < min) ? B0A0A1 : min;
	min = (B1A0A1 < min) ? B1A0A1 : min;
	return min;
}

#undef dot
/* Run for each leaf in parallel */
void FindConflictCells(
	cl_int gid,
	__global OctNode *octree,
	__global Leaf *leaves,
	__global int* nodeToFacet,
	__global Pair *facetBounds,
	__global Line* lines,
	cl_int numLines,
	cl_int keepCollisions,
	__global intn* qpoints,
	cl_int qwidth,
	__global Conflict* conflicts
	);

bool compareConflict(Conflict *a, Conflict *b);
void CompactConflicts(__global Conflict *inputBuffer, __global Conflict *resultBuffer, __global cl_int *predicationBuffer,
	__global cl_int *addressBuffer, cl_int size, const cl_int gid);
#ifndef OpenCL
#undef __local
#undef __global
#else 
#undef cl_int
#undef cl_float
#endif
