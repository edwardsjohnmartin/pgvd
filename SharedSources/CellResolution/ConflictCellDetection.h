#pragma once
#ifndef __OPENCL_VERSION__
#include "../Vector/vec.h"
#include "../Octree/OctNode.h"
#define __local
#define __global
#else
#define cl_int int
#define cl_float float
#include "./SharedSources/Vector/vec.h"
#include "./SharedSources/Octree/OctNode.h"
#endif

//'i' is the indexes of the ambiguous lines
// If the two indexes don't match, 
typedef struct {
    cl_int color;
    cl_int i[2];
    cl_float i2[2];
    cl_int width;
    intn origin;
    //unsigned char padding[8]; //we should see if we can take advantage/remove this padding.
} Conflict;

#ifdef __cplusplus
inline std::ostream& operator<<(std::ostream& os, const Conflict& c) {
    os << "color: "<< c.color << " i:[" <<c.i[0] << ", " << c.i[1] << "]" << " width: " << c.width << " origin: " <<c.origin;
    return os;
}
#endif

/* Ambiguous cells code */
bool liangBarskey(floatn *min, floatn *max, floatn *p1, floatn *p2);

/* Run for each octnode in parallel */
void FindConflictCells(__global OctNode *octree,
    __global FacetPair *facetPairs,
    OctreeData *od,
    __global Conflict* conflicts,
    __global int* nodeToFacet,
    __global Line* lines,
    unsigned int numLines,
    __global intn* points,
    unsigned int gid);

#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#else 
#undef cl_int
#undef cl_float
#endif
