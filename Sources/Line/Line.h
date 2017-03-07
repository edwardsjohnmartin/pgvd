#pragma once
#include "BigUnsigned/BigUnsigned.h"
#include "BinaryRadixTree/BuildBRT.h"
#include "OctreeDefinitions/defs.h"

#ifndef OpenCL
#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include "CL/cl_platform.h"
#endif
#define __local
#define __global
#endif


typedef struct Line {
  cl_int first;
  cl_int second;
  cl_short color;
  cl_short level;
} Line;

/* See paper figure f */
inline void GetLCPFromLine(
  __global Line* lines,
  __global big *zpoints,
  __global LCP *bCells,
  cl_uint mbits,
  cl_uint gid)
{
  cl_int firstI = lines[gid].first;
  cl_int secondI = lines[gid].second;
  big p1 = zpoints[firstI];
  big p2 = zpoints[secondI];
  int lcpLength = compute_lcp_length(&p1, &p2, mbits, gid);
  bCells[gid].len = lcpLength;
  compute_lcp_bu(&bCells[gid].bu, &zpoints[firstI], lcpLength, mbits);
}

#ifndef OpenCL
#undef __local
#undef __global
#endif
