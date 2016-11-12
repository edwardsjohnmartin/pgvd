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
