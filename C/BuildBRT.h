#ifndef __PARALLEL_ALGORITHMS_H__
#define __PARALLEL_ALGORITHMS_H__

#ifndef __OPENCL_VERSION__

#include "../C/BrtNode.h"
#endif

#ifndef __OPENCL_VERSION__ 
	void BuildBinaryRadixTree( BrtNode *I, BrtNode* L, BigUnsigned* mpoints, int mbits, int size, const unsigned int gid);
#else
	void BuildBinaryRadixTree( __global BrtNode *I, __global BrtNode* L, __global BigUnsigned* mpoints, int mbits, int size, const unsigned int gid);
#endif

#endif
