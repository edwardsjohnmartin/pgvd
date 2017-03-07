#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	#include "Dimension/dim.h"
	#include "BigUnsigned/BigNum.h"
#ifdef __cplusplus
}
#endif
#include "Vector/vec.h"

inline big* xyz2z(big* result, intn p, int bits) {
	*result = makeBig(0);
  big temp = makeBig(0);
	big tempb = makeBig(0);
  
  for (int i = 0; i < bits; ++i) {
    //x
    if (p.x & (1 << i)) {
			big temp = makeBig(1);
      temp = shiftBigLeft(&temp, i*DIM + 0);
      *result = orBig(result, &temp);
    }
    //y
    if (p.y & (1 << i)) {
			big temp = makeBig(1);
			temp = shiftBigLeft(&temp, i*DIM + 1);
			*result = orBig(result, &temp);
    }
    //z
#if DIM == 3
    if (p.z & (1 << i)) {
			big temp = makeBig(1);
			temp = shiftBigLeft(&temp, i*DIM + 2);
			*result = orBig(result, &temp);
    }
#endif
  }
  return result;
}
