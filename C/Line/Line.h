#pragma once
#ifndef __OPENCL_VERSION__
  #include "../BigUnsigned/BigUnsigned.h"
  #include "../BinaryRadixTree/BuildBRT.h"
  #define __local
  #define __global
#else
  #include "./OpenCL/C/BigUnsigned/BigUnsigned.h"
  #include "./OpenCL/C/BinaryRadixTree/BuildBRT.h"
#endif

typedef struct Line {
	unsigned int firstIndex;
	unsigned int secondIndex;
  unsigned short color;
  unsigned short level;
  BigUnsigned lcp;
  unsigned int lcpLength;
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
