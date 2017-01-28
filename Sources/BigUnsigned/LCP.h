#pragma once

#include "BigUnsigned/BigUnsigned.h"
#include "OctreeDefinitions/defs.h"

#ifndef OpenCL
#define __global
#define __local
#endif

/* A LCP contains the common bits of two BigUnsigned numbers. 
	 Note that the length of the BU and the LCP length might not match. */
typedef struct LCP {
	BigUnsigned bu;
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

inline int compute_lcp_length(BigUnsigned* a, BigUnsigned* b, cl_int mbits) {
	BigUnsigned tempa, tempb;
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
		shiftBURight(&tempb, b, offset);
		shiftBURight(&tempa, a, offset);

		if (compareBU(&tempa, &tempb) == 0)
			offset -= i;
		else
			offset += i;
	}
	shiftBURight(&tempa, a, offset);
	shiftBURight(&tempb, b, offset);

	if (compareBU(&tempa, &tempb) == 0) {
		shiftBURight(&tempa, a, offset - 1);
		shiftBURight(&tempb, b, offset - 1);
		if (compareBU(&tempa, &tempb) == 0)
			return mbits - (offset - 1);
		else
			return mbits - offset;
	}
	else
		return mbits - (offset + 1);
}

inline void compute_lcp_bu(__global BigUnsigned *lcp_bu, __global BigUnsigned *value, const int length, int mbits) {
	BigUnsigned mask;
	initBlkBU(&mask, 0);
	BigUnsigned one;
	initBlkBU(&one, 1);
	BigUnsigned temp;
	BigUnsigned privateValue, privateLcp;
	privateValue = *value;
	privateLcp = *lcp_bu;
	initBU(&temp);
	for (int i = 0; i < length; ++i) {
		shiftBULeft(&temp, &one, (mbits - 1 - i));
		orBU(&mask, &mask, &temp);
	}
	andBU(&temp, &privateValue, &mask);
	shiftBURight(&privateLcp, &temp, mbits - length);
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
		bool bit1 = getBUBit(&a->bu, i);
		bool bit2 = getBUBit(&b->bu, i);
		if (bit1 == bit2)
			continue;
		else if (bit1 > bit2)
			return 1;
		else return -1;
	}

	//3. Compare blocks one by one from left to right.
	cl_int numBlocks = a->len / 8;
	for (int i = numBlocks - 1; i >= 0; --i) {
		if (getBUBlock(&a->bu, i) == getBUBlock(&b->bu, i))
			continue;
		else if (getBUBlock(&a->bu, i) > getBUBlock(&b->bu, i))
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