#pragma once

#ifdef  __OPENCL_VERSION__ 
  #include "./SharedSources/Dimension/dim.h"
  #include "./SharedSources/BigUnsigned/BigUnsigned.h"
#else
  #include "../../SharedSources/Dimension/dim.h"
  #include "../../SharedSources/BigUnsigned/BigUnsigned.h"
  #include <assert.h>
#endif

// Stores resolution and octree height values
struct Resln {
  // width is the quantized width in one dimension.
  cl_int width;
  cl_int volume;
  // Number of bits per dimension is bits = log(width).
  cl_int bits;
  // Total number of bits for morton code is mbits = bits * DIM.
  cl_int mbits;
};

inline Resln make_resln(const int width_) {
  Resln resln;
  resln.width = width_;
  resln.volume = resln.width;
  for (int i = 1; i < DIM; ++i) {
    resln.volume *= resln.width;
  }
  resln.bits = 0;
  int w = resln.width;
  while (!(w & 1)) {
    ++resln.bits;
    w = w >> 1;
  }
  resln.mbits = resln.bits * DIM;
  return resln;
}
