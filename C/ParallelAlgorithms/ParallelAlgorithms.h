#pragma once

#ifndef __OPENCL_VERSION__
  #include "../BigUnsigned/BigUnsigned.h"
  #include "../Line/Line.h"
  #define __local
  #define __global
#else
  #include "./OpenCL/C/BigUnsigned/BigUnsigned.h"
  #include "./OpenCL/C/Line/Line.h"
#endif

  void GetTwoBitMask(__local BigUnsigned *inputBuffer, __local unsigned int *masks, const unsigned int index, const unsigned char comparedWith, const int lid);
	void BitPredicate( __global BigUnsigned *inputBuffer, __global unsigned int *predicateBuffer, const unsigned int index, const unsigned char comparedWith, const int gid);
  void UniquePredicate(__global BigUnsigned *inputBuffer, __global unsigned int *predicateBuffer, const int gid);
  void LinePredicate(__global Line *inputBuffer, __global unsigned int *predicateBuffer, const unsigned int index, const unsigned char comparedWith, int mbits, const int gid);
  void LevelPredicate(__global Line *inputBuffer, __global unsigned int *predicateBuffer, const unsigned int index, const unsigned char comparedWith, int mbits, const int gid);
  void AddAll(__local unsigned int* localBuffer, const int lid, const int powerOfTwo);
  void HillesSteelScan(__local unsigned int* localBuffer, __local unsigned int* scratch, const int lid, const int powerOfTwo);
  void StreamScan_Init(__global unsigned int* buffer, __local unsigned int* localBuffer, __local unsigned int* scratch, const int gid, const int lid);
  void BUCompact( __global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global unsigned int *lPredicateBuffer, __global unsigned int *leftBuffer, unsigned int size, const int gid);
  void LineCompact(__global Line *inputBuffer, __global Line *resultBuffer, __global unsigned int *lPredicateBuffer, __global unsigned int *leftBuffer, unsigned int size, const int gid);
  void BUSingleCompact( __global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global unsigned int *predicateBuffer, __global unsigned int *addressBuffer, const int gid);
  void StreamScan_SerialKernel(unsigned int* buffer, unsigned int* result, const int size);

#ifndef __OPENCL_VERSION__
  #undef __local
  #undef __global
#endif

