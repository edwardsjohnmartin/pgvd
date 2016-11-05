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
    BigUnsigned lcp;
    cl_int lcpLength;
} Line;

inline void calculateLineLCP(__global Line* lines, __global BigUnsigned *zpoints, unsigned int mbits, unsigned int gid) {
    BigUnsigned first = zpoints[lines[gid].firstIndex];
    BigUnsigned second = zpoints[lines[gid].secondIndex];
    int lcpLength = compute_lcp_length(&first, &second, mbits);
    BigUnsigned lcp;
    compute_lcp(&lines[gid].lcp, &zpoints[lines[gid].firstIndex], lcpLength, mbits);
    lines[gid].lcpLength = lcpLength;
}

#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif
