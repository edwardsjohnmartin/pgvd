#pragma once

#ifdef __OPENCL_VERSION__
	#include "./OpenCL/C/BinaryRadixTree/BrtNode.h"
#else
	#include "BrtNode.h"
	#define __global
	#define __local
#endif

void BuildBinaryRadixTree( __global BrtNode *I, __global BigUnsigned* mpoints, int mbits, int size, const unsigned int gid);
void compute_lcp(__global BigUnsigned *lcp, __global BigUnsigned *value, const int length, int mbits);
int compute_lcp_length(BigUnsigned* a, BigUnsigned* b, int mbits);
int compareLCP(BigUnsigned *a, BigUnsigned *b, unsigned int a_len, unsigned int b_len);

#ifndef __OPENCL_VERSION__
	#undef __local
	#undef __global
#endif
