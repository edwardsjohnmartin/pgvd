#pragma once
#include "OctreeDefinitions/defs.h"

#ifndef OpenCL
#include <stdbool.h>
#include <stdio.h>
#endif

#ifndef NULL
#define NULL 0
#endif

//#define NumBlocks 1
#define NumBlocks 2
//#define NumBlocks 4
//#define NumBlocks 8

#define NumBytes (NumBlocks * sizeof(cl_ulong))
#define NumBitsPerBlock (sizeof(cl_ulong) * 8)
#define BLOCKMAX 18446744073709551615

typedef struct big {
	cl_ulong blk[NumBlocks];
} big;

//~~INITIALIZERS~~//
inline big copyBig(big x) {
	big result;
#pragma unroll
	for (int i = 0; i < NumBlocks; i++) result.blk[i] = x.blk[i];
	return result;
}
inline big makeBig(cl_ulong x) {
	big result; 
	result.blk[0] = x;
#pragma unroll
	for (int i = 1; i < NumBlocks; ++i) result.blk[i] = 0;
	return result;
}

inline big makeMaxBig() {
	big result;
#pragma unroll
	for (int i = 0; i < NumBlocks; ++i) {
		result.blk[i] = BLOCKMAX;
	}
	return result;
}

//~~BIT/BLOCK ACCESSORS~~//
static inline cl_ulong getLShiftedBlk(big *num, cl_uint blkIndx, cl_uint shift) {
	cl_ulong part1 = (blkIndx == 0 || shift == 0) ? 0 : (num->blk[blkIndx - 1] >> (NumBitsPerBlock - shift));
	cl_ulong part2 = num->blk[blkIndx] << shift;
	return part1 | part2;
}
static inline cl_ulong getRShiftedBlk(big *num, cl_uint blkIndx, cl_uint shift) {
	cl_ulong part1 = (blkIndx == NumBlocks - 1 || shift == 0) ? 0 : (num->blk[blkIndx + 1] << (NumBitsPerBlock - shift));
	cl_ulong part2 = num->blk[blkIndx] >> shift;
	return part1 | part2;
}
static inline cl_uchar getBigBit(big *b, cl_uint blkIndx, cl_uint bitIndx) {
	return (b->blk[blkIndx] & (1UL << bitIndx)) != 0UL;
}
inline void setBigBit(big *bn, cl_uint blkIndx, cl_uint bitIndx, cl_uchar newBit) {
	cl_ulong block = bn->blk[blkIndx], mask = 1UL << bitIndx;
	block = newBit ? (block | mask) : (block & ~mask);
	bn->blk[blkIndx] = block;
}

//~~COMPARISON~~//
inline int compareBig(big *x, big *y) {
#pragma unroll
	for (int i = NumBlocks - 1; i >= 0; --i) {
		if (x->blk[i] > y->blk[i]) return 1;
		if (x->blk[i] < y->blk[i]) return -1;
	}
	return 0;
}
inline int weakCompareBig(big x, big y) {
#pragma unroll
	for (int i = NumBlocks - 1; i >= 0; --i) {
		if (x.blk[i] > y.blk[i]) return 1;
		if (x.blk[i] < y.blk[i]) return 0;
	}
	return 0;
}
inline bool weakEqualsBig(big x, big y) {
#pragma unroll
	for (int i = NumBlocks - 1; i >= 0; --i) {
		if (x.blk[i] > y.blk[i]) return 0;
		if (x.blk[i] < y.blk[i]) return 0;
	}
	return 1;
}

//~~ARITHMATIC OPERATIONS~~//
inline big addBig(big *a, big *b) {
	// Carries in and out of an addition stage
	big result = makeBig(0);
	bool carryOut;
	cl_ulong temp;

#pragma unroll
	for (int i = 0, carryIn = false; i < NumBlocks; ++i) {
		temp = a->blk[i] + b->blk[i];
		//If rollover occured, temp is less than either input.
		carryOut = temp < a->blk[i];
		if (carryIn) {
			temp++;
			carryOut |= (temp == 0);
		}
		result.blk[i] = temp;
		carryIn = carryOut;
	}
	return result;
}
inline big subtractBig(big *a, big *b) {
	// some variables...
	big result = makeBig(0);
	bool borrowin, borrowout;
	cl_ulong temp;

#pragma unroll
	for (int i = 0, borrowin = false; i < NumBlocks; ++i) {
		temp = a->blk[i] - b->blk[i];

		//if reverse rollover occured, temp is greater than block a
		borrowout = temp > a->blk[i];
		if (borrowin) {
			borrowout |= (temp == 0);
			temp--;
		}
		result.blk[i] = temp;
		borrowin = borrowout;
	}
	return result;
}

//~~BITWISE OPERATORS~~//
/* These are straightforward blockwise operations except that they differ in
* the output length and the necessity of zapLeadingZeros. */
inline big andBig(big *a, big *b) {
	big result;
#pragma unroll
	for (int i = 0; i < NumBlocks; ++i)
		result.blk[i] = a->blk[i] & b->blk[i];
	return result;
}
inline big orBig(big *a, big *b) {
	big result;
#pragma unroll
	for (int i = 0; i < NumBlocks; ++i)
		result.blk[i] = a->blk[i] | b->blk[i];
	return result;
}
inline big xOrBig(big *a, big *b) {
	big result;
#pragma unroll
	for (int i = 0; i < NumBlocks; ++i)
		result.blk[i] = a->blk[i] ^ b->blk[i];
	return result;
}

inline big shiftBigRight(big *a, cl_uint b) {
	big result = makeBig(0);
	cl_int blksToSkip = b / NumBitsPerBlock;
	cl_int shift = b % NumBitsPerBlock;

	for (cl_int j = NumBlocks - 1, i = NumBlocks - blksToSkip - 1; i >= 0; j--, i--)
		result.blk[i] = getRShiftedBlk(a, j, shift);

	return result;
}
inline big shiftBigLeft(big *a, unsigned b) {
	big result = makeBig(0);
	cl_int shiftBlocks = b / NumBitsPerBlock;
	cl_int shiftBits = b % NumBitsPerBlock;

	for (cl_int j = 0, i = shiftBlocks; i < NumBlocks; j++, i++)
		result.blk[i] = getLShiftedBlk(a, j, shiftBits);

	return result;
}
