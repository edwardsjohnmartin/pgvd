#pragma once
#ifndef __OPENCL_VERSION__
#include "../BigUnsigned/BigUnsigned.h"
#include "../BinaryRadixTree/BuildBRT.h"
#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include "CL/cl_platform.h"
#endif
#define __local
#define __global
#else
#include "./SharedSources/OctreeDefinitions/defs.h"
#include "./SharedSources/BigUnsigned/BigUnsigned.h"
#include "./SharedSources/BinaryRadixTree/BuildBRT.h"
#endif


typedef struct Line {
    cl_int firstIndex;
    cl_int secondIndex;
    cl_short color;
    cl_short level;
} Line;

//EnqueueFillBuffer requires power of two...
typedef struct BCell {
    BigUnsigned lcp;
    cl_int lcpLength;
    cl_int padding[3];
} BCell;

/* See paper figure f */
inline void GetBCellLCP(
    __global Line* lines, 
    __global BigUnsigned *zpoints, 
    __global BCell *bCells,
    __global cl_int *facetIndices,
    cl_uint mbits, 
    cl_uint gid) 
{
    cl_int firstI = lines[gid].firstIndex;
    cl_int secondI = lines[gid].secondIndex;
    BigUnsigned p1 = zpoints[firstI];
    BigUnsigned p2= zpoints[secondI];
    facetIndices[gid] = gid;
    int lcpLength = compute_lcp_length(&p1, &p2, mbits);
    bCells[gid].lcpLength = lcpLength;
    compute_lcp(&bCells[gid].lcp, &zpoints[firstI], lcpLength, mbits);
}

#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif
