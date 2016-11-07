#pragma once

#ifndef __OPENCL_VERSION__
#include "../BigUnsigned/BigUnsigned.h"
#include "../Line/Line.h"
#define __local
#define __global
#else
#include "./SharedSources/BigUnsigned/BigUnsigned.h"
#include "./SharedSources/Line/Line.h"
#endif

void GetTwoBitMask(__local BigUnsigned *inputBuffer, __local unsigned int *masks, const unsigned int index, const unsigned char comparedWith, const int lid);
void BitPredicate(__global int *inputBuffer, __global unsigned int *predicateBuffer, const unsigned int index, const unsigned char comparedWith, const int gid);
void BUBitPredicate(__global BigUnsigned *inputBuffer, __global unsigned int *predicateBuffer, const unsigned int index, const unsigned char comparedWith, const int gid);
void UniquePredicate(__global BigUnsigned *inputBuffer, __global unsigned int *predicateBuffer, const int gid);
void BCellPredicate(__global BCell *inputBuffer, __global unsigned int *predicateBuffer, const unsigned int index, const unsigned char comparedWith, int mbits, const int gid);
void LevelPredicate(__global BCell *inputBuffer, __global unsigned int *predicateBuffer, const unsigned int index, const unsigned char comparedWith, int mbits, const int gid);
void AddAll(__local unsigned int* localBuffer, const int lid, const int powerOfTwo);
void HillesSteelScan(__local unsigned int* localBuffer, __local unsigned int* scratch, const int lid, const int powerOfTwo);
void StreamScan_Init(__global unsigned int* buffer, __local unsigned int* localBuffer, __local unsigned int* scratch, const int gid, const int lid);
void BUCompact(__global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global unsigned int *lPredicateBuffer, __global unsigned int *leftBuffer, unsigned int size, const int gid);
void Compact(__global int *inputBuffer, __global int *resultBuffer, __global unsigned int *lPredicateBuffer,
    __global unsigned int *leftBuffer, unsigned int size, const int gid);
void BCellFacetCompact(__global BCell *inputBCellBuffer,
    __global cl_int *inputIndexBuffer,
    __global BCell *resultBCellBuffer,
    __global cl_int *resultIndexBuffer,
    __global unsigned int *lPredicateBuffer,
    __global unsigned int *leftBuffer,
    unsigned int size,
    const int gid);
void BUSingleCompact(__global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global unsigned int *predicateBuffer, __global unsigned int *addressBuffer, const int gid);
void StreamScan_SerialKernel(unsigned int* buffer, unsigned int* result, const int size);

#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif

