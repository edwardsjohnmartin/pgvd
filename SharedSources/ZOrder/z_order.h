#pragma once
#ifdef __OPENCL_VERSION__
  #include "./SharedSources/BigUnsigned/BigUnsigned.h"
  #include "./SharedSources/Vector/vec.h"
  #include "./SharedSources/Dimension/dim.h"
#elif defined __cplusplus
extern "C" {
	#include "../../SharedSources/Dimension/dim.h"
	#include "../../SharedSources/BigUnsigned/BigUnsigned.h"
}
  #include "../../SharedSources/Vector/vec.h"
#else
#error Only C++/OpenCL can compile z_order.h!
#endif

inline BigUnsigned* xyz2z(BigUnsigned *result, intn p, int bits) {
  initBlkBU(result, 0);
  BigUnsigned temp;
  initBlkBU(&temp, 0);
  BigUnsigned tempb;
  initBlkBU(&tempb, 0);
  
  for (int i = 0; i < bits; ++i) {
    //x
    if (p.x & (1 << i)) {
      initBlkBU(&temp, 1);
      shiftBULeft(&tempb, &temp, i*DIM + 0);
      initBUBU(&temp, result);
      orBU(result, &temp, &tempb);
    }
    //y
    if (p.y & (1 << i)) {
      initBlkBU(&temp, 1);
      shiftBULeft(&tempb, &temp, i*DIM + 1);
      initBUBU(&temp, result);
      orBU(result, &temp, &tempb);
    }
    //z
#if DIM == 3
    if (p.z & (1 << i)) {
      initBlkBU(&temp, 1);
      shiftBULeft(&tempb, &temp, i*DIM + 2);
      initBUBU(&temp, result);
      orBU(result, &temp, &tempb);
    }
#endif
  }
  return result;
}
