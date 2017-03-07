#pragma once

#include "BigUnsigned/BigNum.h"
#include "Line/Line.h"

#ifndef OpenCL
#define __local
#define __global
#endif

void GetTwoBitMask(__local big *inputBuffer, __local cl_int *masks, const cl_int index, const char comparedWith, const cl_int lid);
void BitPredicate(__global cl_int *inputBuffer, __global cl_int *predicateBuffer, cl_int index, cl_int comparedWith, cl_int gid);
void BitPredicateULL(__global unsigned long long *inputBuffer, __global cl_int *predicateBuffer, cl_int index, cl_int comparedWith, cl_int gid);
void BigBitPredicate(__global big *inputBuffer, __global cl_int *predicateBuffer, cl_int index, cl_int comparedWith, cl_int gid);
void BigUniquePredicate(__global big *inputBuffer, __global cl_int *predicateBuffer, const cl_int gid);
void LCPPredicate(__global LCP *inputBuffer, __global cl_int *predicateBuffer, cl_int index, cl_int comparedWith, cl_int mbits, cl_int gid);
void LevelPredicate(__global LCP *inputBuffer, __global cl_int *predicateBuffer, const cl_int index, const unsigned char comparedWith, cl_int mbits, const cl_int gid);
void BigCompact(__global big *inputBuffer, __global big *resultBuffer, __global cl_int *lPredicateBuffer, __global cl_int *leftBuffer, cl_int size, const cl_int id);
void Compact(__global cl_int *inputBuffer, __global cl_int *resultBuffer, __global cl_int *predicationBuffer,
  __global cl_int *addressBuffer, cl_int size, const cl_int gid);
void CompactULL(__global unsigned long long *inputBuffer, __global unsigned long long *resultBuffer, __global cl_int *predicationBuffer,
	__global cl_int *addressBuffer, cl_int size, const cl_int gid);
void LCPFacetCompact(__global LCP *inputBCellBuffer,
    __global cl_int *inputIndexBuffer,
    __global LCP *resultBCellBuffer,
    __global cl_int *resultIndexBuffer,
    __global cl_int *lPredicateBuffer,
    __global cl_int *leftBuffer,
		cl_int size,
    const cl_int gid);
void BigSingleCompact(__global big *inputBuffer, __global big *resultBuffer, __global cl_int *predicateBuffer, __global cl_int *addressBuffer, const cl_int gid);

#ifndef OpenCL
#undef __local
#undef __global
#endif

