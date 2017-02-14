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
	cl_int x = (((self & (1 << index)) >> index) == comparedWith);
	predicateBuffer[gid] = x;
}

void BUBitPredicate(__global BigUnsigned *inputBuffer, __global cl_int *predicateBuffer, cl_int index, cl_int comparedWith, cl_int gid)
{
  BigUnsigned self = inputBuffer[gid];
  cl_int x = (getBUBit(&self, index) == comparedWith);
  predicateBuffer[gid] = x;
}

void GetTwoBitMask(
  __local BigUnsigned *inputBuffer,
  __local cl_int *masks,
  const cl_int index,
  const char comparedWith,
  const cl_int lid)
{
  BigUnsigned self;
  unsigned char x = 0;
  cl_int offset = lid * 4;

  masks[offset] = masks[offset + 1] = masks[offset + 2] = masks[offset + 3] = 0;
  self = inputBuffer[lid];
  cl_int numberBUBits = self.len * sizeof(Blk) * 8;
  if (numberBUBits > index)
    x = (getBUBit(&self, index) == comparedWith);
  if (numberBUBits > index + 1)
    x |= (getBUBit(&self, index + 1) == comparedWith) << 1;

#ifdef OpenCL
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
  masks[offset + x] = 1;
}

//Unique Predication
//Requires input be sorted.
void BUUniquePredicate(
  __global BigUnsigned *inputBuffer,
  __global cl_int *predicateBuffer,
  const cl_int gid)
{
  if (gid == 0) {
    predicateBuffer[gid] = 1;
  }
  else {
    BigUnsigned self = inputBuffer[gid];
    BigUnsigned previous = inputBuffer[gid - 1];
    predicateBuffer[gid] = (compareBU(&self, &previous) != 0);
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
    bool myBit = getBUBit(&bCell.bu, actualIndex);
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
void BUCompact(__global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global cl_int *lPredicateBuffer,
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


void BUSingleCompact(__global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global cl_int *predicateBuffer, __global cl_int *addressBuffer, const cl_int gid)
{
  cl_int index;
  if (predicateBuffer[gid] == 1) {
    index = addressBuffer[gid];
    BigUnsigned temp = inputBuffer[gid];
    resultBuffer[index - 1] = temp;
  }
}

#ifndef OpenCL
#undef __local
#undef __global
#endif
