#pragma once
#ifdef __OPENCL_VERSION__
  #include "./OpenCL/C/BigUnsigned/BigUnsigned.h"
  #include "./OpenCL/C/Vector/vec_n.h"
  #include "./OpenCL/C/Dimension/dim.h"
#else
  #include "../../C/Dimension/dim.h"
  #include "../../C/BigUnsigned/BigUnsigned.h"
  #include "../../C/Vector/vec_n.h"
#endif // !__OPENCL_VERSION__

inline BigUnsigned* xyz2z(BigUnsigned *result, int_n p, int bits) {
  initBlkBU(result, 0);
  BigUnsigned temp;
  initBlkBU(&temp, 0);
  BigUnsigned tempb;
  initBlkBU(&tempb, 0);
  
  for (int i = 0; i < bits; ++i) {
    //x
    if (X_(p) & (1 << i)) {
      initBlkBU(&temp, 1);
      shiftBULeft(&tempb, &temp, i*DIM + 0);
      initBUBU(&temp, result);
      orBU(result, &temp, &tempb);
    }
    //y
    if (Y_(p) & (1 << i)) {
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
