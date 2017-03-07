#include "ParallelAlgorithms/ParallelAlgorithms.h"
#include "Dimension/dim.h"
#ifndef OpenCL
#define __local
#define __global
#endif

//If the bit at the provided cl_int matches compared with, the predicate buffer at n is set to 1. 0 otherwise.
void BitPredicate(__global cl_int *inputBuffer, __global cl_int *predicateBuffer, cl_int index, cl_int comparedWith, cl_int gid)
{
  cl_int self = inputBuffer[gid];
  cl_int x = (((self & (1 << index)) >> index) == comparedWith);
  predicateBuffer[gid] = x;
}

void BitPredicateULL(__global unsigned long long *inputBuffer, __global cl_int *predicateBuffer, cl_int index, cl_int comparedWith, cl_int gid)
{
	unsigned long long self = inputBuffer[gid];
	cl_int x = (((self & (1UL << index)) >> index) == comparedWith);
	predicateBuffer[gid] = x;
}

void BigBitPredicate(__global big *inputBuffer, __global cl_int *predicateBuffer, cl_int index, cl_int comparedWith, cl_int gid)
{
  big self = inputBuffer[gid];
	cl_int bit = index % NumBitsPerBlock;
	cl_int blk = index / NumBitsPerBlock;
  cl_int x = (getBigBit(&self, blk, bit) == comparedWith);
  predicateBuffer[gid] = x;
}

//Unique Predication
//Requires input be sorted.
//TODO: CHANGE TO USE SHARED MEMORY
void BigUniquePredicate(
  __global big *inputBuffer,
  __global cl_int *predicateBuffer,
  const cl_int gid)
{
  if (gid == 0) {
    predicateBuffer[gid] = 1;
  }
  else {
    big self = inputBuffer[gid];
    big previous = inputBuffer[gid - 1];
    predicateBuffer[gid] = (compareBig(&self, &previous) != 0);
  }
}

void LCPPredicate(
  __global LCP *inputBuffer,
  __global cl_int *predicateBuffer,
  cl_int index,
  cl_int comparedWith,
  cl_int mbits,
  cl_int gid)
{
  LCP bCell = inputBuffer[gid];
  unsigned lcpLength = bCell.len;
  int actualIndex = index - (mbits - lcpLength);
  int shift = lcpLength % DIM;
  if (actualIndex - shift >= 0) {
		cl_int bit = actualIndex % NumBitsPerBlock;
		cl_int blk = actualIndex / NumBitsPerBlock;
    bool myBit = getBigBit(&bCell.bu, blk, bit);
    predicateBuffer[gid] = myBit == 0;
  }
  else {
    predicateBuffer[gid] = 1;// !comparedWith;
  }
}

void LevelPredicate(__global LCP *inputBuffer, __global cl_int *predicateBuffer, const cl_int index, const unsigned char comparedWith, cl_int mbits, const cl_int gid) {
  LCP bCell = inputBuffer[gid];
	cl_int lcpLength = bCell.len;
	cl_int level = lcpLength / DIM;
	cl_int maxLevel = (mbits / DIM) - 1;
  predicateBuffer[gid] = (maxLevel - index < level) == comparedWith;
}

//result buffer MUST be initialized as 0!!!
void BigCompact(__global big *inputBuffer, __global big *resultBuffer, __global cl_int *lPredicateBuffer,
  __global cl_int *leftBuffer, cl_int size, const cl_int id)
{
	cl_int a = leftBuffer[id];
	cl_int b = leftBuffer[size - 2];
  cl_int c = lPredicateBuffer[id];
  cl_int e = lPredicateBuffer[size - 1];
    
  //Check out http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html figure 39-14
  cl_int t = id - a + (e + b);
  cl_int d = (!c) ? t : a - 1;

  //This suffers a bit from poor coalescing
  resultBuffer[d] = inputBuffer[id];
}

void Compact(__global cl_int *inputBuffer, __global cl_int *resultBuffer, __global cl_int *predicationBuffer,
  __global cl_int *addressBuffer, cl_int size, const cl_int gid)
{
  cl_int a = addressBuffer[gid];
  cl_int b = addressBuffer[size - 2];
  cl_int c = predicationBuffer[gid];
  cl_int e = predicationBuffer[size - 1];

  //Check out http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html figure 39-14
  cl_int t = gid - a + (e + b);
  cl_int d = (!c) ? t : a - 1;

  //This really suffers from poor coalescing
#ifdef OpenCL
  barrier(CLK_GLOBAL_MEM_FENCE);
#endif
  resultBuffer[d] = inputBuffer[gid];
}

void CompactULL(__global unsigned long long *inputBuffer, __global unsigned long long *resultBuffer, __global cl_int *predicationBuffer,
	__global cl_int *addressBuffer, cl_int size, const cl_int gid)
{
	cl_int a = addressBuffer[gid];
	cl_int b = addressBuffer[size - 2];
	cl_int c = predicationBuffer[gid];
	cl_int e = predicationBuffer[size - 1];

	//Check out http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html figure 39-14
	cl_int t = gid - a + (e + b);
	cl_int d = (!c) ? t : a - 1;

#ifdef OpenCL
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	resultBuffer[d] = inputBuffer[gid];
}

//result buffer MUST be initialized as 0!!!
void LCPFacetCompact(
  __global LCP *inputBCellBuffer,
  __global cl_int *inputIndexBuffer,
  __global LCP *resultBCellBuffer,
  __global cl_int *resultIndexBuffer,
  __global cl_int *lPredicateBuffer,
  __global cl_int *leftBuffer,
	cl_int size,
  const cl_int gid)
{
	cl_int a = leftBuffer[gid];
	cl_int b = leftBuffer[size - 2];
	cl_int c = lPredicateBuffer[gid];
	cl_int e = lPredicateBuffer[size - 1];

  //Check out http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html figure 39-14
	cl_int t = gid - a + (e + b);
	cl_int d = (!c) ? t : a - 1;

  //This really suffers from poor coalescing
#ifdef OpenCL
  barrier(CLK_GLOBAL_MEM_FENCE);
#endif
  resultBCellBuffer[d] = inputBCellBuffer[gid];
  resultIndexBuffer[d] = inputIndexBuffer[gid];
}


void BigSingleCompact(__global big *inputBuffer, __global big *resultBuffer, __global cl_int *predicateBuffer, __global cl_int *addressBuffer, const cl_int gid)
{
  cl_int index;
  if (predicateBuffer[gid] == 1) {
    index = addressBuffer[gid];
    big temp = inputBuffer[gid];
    resultBuffer[index - 1] = temp;
  }
}

#ifndef OpenCL
#undef __local
#undef __global
#endif
