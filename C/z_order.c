/* #ifndef __OPENCL_VERSION__ */
/* #include "z_order.h"  */
/* #endif  */

/* BigUnsigned* xyz2z(BigUnsigned *result, intn p, int bits) { */
/*   initBlkBU(result, 0); */
/*   BigUnsigned temp; */
/*   initBlkBU(&temp, 0); */
/*   BigUnsigned tempb; */
/*   initBlkBU(&tempb, 0); */
  
/*   for (int i = 0; i < bits; ++i) { */
/*     for (int j = 0; j < DIM; ++j) { */
/* #ifdef __OPENCL_VERSION__ */
/*       if (p[j] & (1 << i)) { */
/* #else */
/*       if (p.s[j] & (1 << i)) { */
/* #endif */
/*         //ret |= BigUnsigned(1) << (i*DIM + j); */
/*         initBlkBU(&temp, 1); */
/*         shiftBULeft(&tempb, &temp, i*DIM + j); */
/*         initBUBU(&temp, result); */
/*         orBU(result, &temp, &tempb); */
/*       } */
/*     } */
/*   } */
/*   return result; */
/* } */
