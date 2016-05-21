#ifndef __PARALLEL_ALGORITHMS_H__
#define __PARALLEL_ALGORITHMS_H__

#ifndef __OPENCL_VERSION__
#include "BigUnsigned.h"
#endif

#ifndef __OPENCL_VERSION__
#define __local
#define __global
#endif

	void BitPredicate( __global BigUnsigned *inputBuffer, __global Index *predicateBuffer, const Index index, const unsigned char comparedWith, const int gid);
	void UniquePredicate( __global BigUnsigned *inputBuffer, __global Index *predicateBuffer, const size_t gid);
  void AddAll(__local Index* localBuffer, const int lid, const int powerOfTwo);
  void HillesSteelScan(__local Index* localBuffer, __local Index* scratch, const int lid, const int powerOfTwo);
  void StreamScan_Init(__global Index* buffer, __local Index* localBuffer, __local Index* scratch, const int gid, const int lid);
  void BUCompact( __global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global Index *lPredicateBuffer, __global Index *leftBuffer, Index size, const size_t gid);
	void BUSingleCompact( __global BigUnsigned *inputBuffer, __global BigUnsigned *resultBuffer, __global Index *predicateBuffer, __global Index *addressBuffer, const int gid);
  void StreamScan_SerialKernel(Index* buffer, Index* result, const int size);

#endif
