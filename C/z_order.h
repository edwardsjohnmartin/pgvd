#ifndef __OPENCL_VERSION__
extern "C" {
  #include "BigUnsigned.h"
}
#include "../opencl/vec.h"
#define DIM 2
#endif // !__OPENCL_VERSION__

BigUnsigned* xyz2z(BigUnsigned *result, intn p, int bits);
