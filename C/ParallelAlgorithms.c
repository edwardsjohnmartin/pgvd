#ifdef __OPENCL_VERSION__
#include "opencl\C\ParallelAlgorithms.h"
#else
#include "ParallelAlgorithms.h"
#endif

//If the bit at the provided index matches compared with, the predicate buffer at n is set to 1. 0 otherwise.
void BitPredicate( __global BigUnsigned *inputBuffer, __global Index *predicateBuffer, const Index index, const unsigned char comparedWith, const int gid)
{
  BigUnsigned self = inputBuffer[gid];
  predicateBuffer[gid] = (getBUBit(&self, index) == comparedWith) ? 1:0;
}

//Unique Predication
//Requires input be sorted.
void UniquePredicate( __global BigUnsigned *inputBuffer, __global Index *predicateBuffer, const int gid)
{
  if (gid == 0) {
    predicateBuffer[gid] = 1;
  } else {
    BigUnsigned self = inputBuffer[gid];
    BigUnsigned previous = inputBuffer[gid-1];
    predicateBuffer[gid] = (compareBU(&self, &previous) != 0);
  }
}

void StreamScan_Init(__global Index* buffer, __local Index* localBuffer, __local Index* scratch, const int gid, const int lid)
{
  localBuffer[lid] = scratch[lid] = buffer[gid];
}

void AddAll(__local Index* localBuffer, const int lid, const int powerOfTwo)
{
    if (lid < powerOfTwo) {
      localBuffer[lid] = localBuffer[lid + powerOfTwo] + localBuffer[lid];
    }
}

void HillesSteelScan(__local Index* localBuffer, __local Index* scratch, const int lid, const int powerOfTwo)
{
    if (lid > (powerOfTwo - 1))
      scratch[lid] = localBuffer[lid] + localBuffer[lid - powerOfTwo];
    else
      scratch[lid] = localBuffer[lid];
}

//result buffer MUST be initialized as 0!!!
void BUCompact( __global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global Index *lPredicateBuffer, 
	__global Index *leftBuffer, Index size, const int gid)
{
  //Check out http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html figure 39-14
  int t = gid - leftBuffer[gid] + (lPredicateBuffer[size - 1] + leftBuffer[size - 2]);
  int d = (!lPredicateBuffer[gid]) ? t : leftBuffer[gid] - 1;
  resultBuffer[d] = inputBuffer[gid];
}

void BUSingleCompact( __global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global Index *predicateBuffer, __global Index *addressBuffer, const int gid)
{
  Index index;
  if (predicateBuffer[gid] == 1) {
    index = addressBuffer[gid];
    BigUnsigned temp = inputBuffer[gid];
    resultBuffer[index - 1] = temp;
  } 
}

#ifndef __OPENCL_VERSION__
  #include <stdlib.h>
  #include <stdio.h>
  #include <math.h>
  void StreamScan_SerialKernel(Index* buffer, Index* result, const int size) {
    int nextPowerOfTwo = (int)pow(2, ceil(log(size) / log(2)));
	  int intermediate = -1;
	  Index* localBuffer;
	  Index* scratch;
    Index sum = 0;

	  localBuffer = (Index*) malloc(sizeof(Index)* nextPowerOfTwo);
	  scratch = (Index*) malloc(sizeof(Index)* nextPowerOfTwo);
    //INIT
    for (int i = 0; i < size; i++)
      StreamScan_Init(buffer, localBuffer, scratch, i, i);
    for (int i = size; i < nextPowerOfTwo; ++i) 
      localBuffer[i] = scratch[i] = 0;
    
    //Add not necessary with only one workgroup.
    //Adjacent sync not necessary with only one workgroup.

    //SCAN
    for (unsigned int i = 1; i < nextPowerOfTwo; i <<= 1) {
      for (int j = 0; j < nextPowerOfTwo; ++j) {
        HillesSteelScan(localBuffer, scratch, j, i);
      }
      __local Index *tmp = scratch;
      scratch = localBuffer;
      localBuffer = tmp;
    }
    for (int i = 0; i < size; ++i) {
      result[i] = localBuffer[i];
    }
	  free(localBuffer);
	  free(scratch);
  }
#endif
