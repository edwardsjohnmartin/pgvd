#ifndef __PARALLEL_ALGORITHMS_H__
#define __PARALLEL_ALGORITHMS_H__

#ifndef __OPENCL_VERSION__

#include "../C/BrtNode.h"
#define __global
#define __local
#endif
typedef unsigned int size_t;

void BuildBinaryRadixTree( __global BrtNode *I, __global BrtNode* L, __global BigUnsigned* mpoints, int mbits, int size, const unsigned int gid);
void compute_lcp(__global BigUnsigned *lcp, __global BigUnsigned *value, const int length, int mbits);
int compute_index_lcp_length(size_t i, size_t j);
int compute_lcp_length(size_t i, size_t j, __global BigUnsigned* _mpoints, int mbits);

#endif
