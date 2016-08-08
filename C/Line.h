#pragma once
#ifndef __OPENCL_VERSION__
#include "BigUnsigned.h"
#include "BuildBRT.h"
#else
#include "./opencl/C/BigUnsigned.h"
#include "./opencl/C/BuildBRT.h"
#endif

#ifndef __OPENCL_VERSION__
#define __local
#define __global
#endif

typedef struct Line {
	unsigned int firstIndex;
	unsigned int secondIndex;
  unsigned int color;
  BigUnsigned lcp;
  int lcpLength;
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
