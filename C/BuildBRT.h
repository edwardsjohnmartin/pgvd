#ifndef __BUILD_BRT_H__
#define __BUILD_BRT_H__

#ifdef __OPENCL_VERSION__
#include "./opencl/C/BrtNode.h"
#else
#include "../C/BrtNode.h"
#define __global
#define __local
#endif

void BuildBinaryRadixTree( __global BrtNode *I, __global BigUnsigned* mpoints, int mbits, int size, const unsigned int gid);
void compute_lcp(__global BigUnsigned *lcp, __global BigUnsigned *value, const int length, int mbits);
int compute_lcp_length(BigUnsigned* a, BigUnsigned* b, int mbits);

#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif
#endif
