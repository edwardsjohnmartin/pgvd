#pragma once

#include "BigUnsigned/BigNum.h"
#include "OctreeDefinitions/defs.h"

#ifndef OpenCL
#define __global
#define __local
#endif

/* A LCP contains the common bits of two big numbers. 
	 Note that the length of the BU and the LCP length might not match. */
typedef struct LCP {
	big bu;
	cl_int len;
} LCP;

// LEAST COMMON PREFIX CALCULATIONS (\delta in karras2014)
// Longest common prefix
//
// Suppose mbits = 6, then morton code is
//   ______
// 00011010
//
// Suppose length = 3, then lcp (masked) is
//   ___
// 00011000
//
// Now shift, and lcp is
//      ___
// 00000011
#ifndef OpenCL
#include <inttypes.h>
#endif
inline int compute_lcp_length(big* a, big* b, cl_int mbits, cl_int temp) {
	big tempa, tempb;
	cl_int v = mbits; // compute the next highest power of 2 of 32-bit v
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;

	cl_int offset = v >> 1;
	for (cl_int i = v >> 2; i > 0; i >>= 1) {
		tempa = shiftBigRight(a, offset);
		tempb = shiftBigRight(b, offset);
		if (compareBig(&tempa, &tempb) == 0)
			offset -= i;
		else
			offset += i;
	}

	tempa = shiftBigRight(a, offset);
	tempb = shiftBigRight(b, offset);

	if (compareBig(&tempa, &tempb) == 0) {
		tempa = shiftBigRight(a, offset - 1);
		tempb = shiftBigRight(b, offset - 1);
		if (compareBig(&tempa, &tempb) == 0)
			return mbits - (offset - 1);
		else
			return mbits - offset;
	}
	else
		return mbits - (offset + 1);
}

inline void compute_lcp_bu(__global big *lcp_bu, __global big *value, const int length, int mbits) {
	big mask = makeBig(0);
	big one = makeBig(1);
	big temp = makeBig(0);
	big privateValue, privateLcp;
	privateValue = *value;
	privateLcp = *lcp_bu;
	for (int i = 0; i < length; ++i) {
		temp = shiftBigLeft(&one, (mbits - 1 - i));
		mask = orBig(&mask, &temp);
	}
	temp = andBig(&privateValue, &mask);
	privateLcp = shiftBigRight(&temp, mbits - length);
	*value = privateValue;
	*lcp_bu = privateLcp;
}

inline int compareLCP(LCP *a, LCP *b) {
	//1. Compare length.
	if (a->len > b->len) return 1;
	if (a->len < b->len) return -1;

	//2. Compare bits of the block after last
	cl_int numBits = a->len % 8;
	for (cl_int i = a->len - 1; i >= a->len - numBits; --i) {
		cl_int blk = i % NumBitsPerBlock;
		cl_int bit = i / NumBitsPerBlock;
		bool bit1 = getBigBit(&a->bu, blk, bit);
		bool bit2 = getBigBit(&b->bu, blk, bit);
		if (bit1 == bit2)
			continue;
		else if (bit1 > bit2)
			return 1;
		else return -1;
	}

	//3. Compare blocks one by one from left to right.
	cl_int numBlocks = a->len / 8;
	for (int i = numBlocks - 1; i >= 0; --i) {
		if (a->bu.blk[i] == b->bu.blk[i])
			continue;
		else if (a->bu.blk[i] > b->bu.blk[i])
			return 1;
		else
			return -1;
	}

	//4. The two LCPS are equivalent.
	return 0;
}

#ifndef OpenCL
#undef __local
#undef __global
#endif