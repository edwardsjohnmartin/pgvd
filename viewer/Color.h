#ifndef __COLOR_H__
#define __COLOR_H__

#include "../opencl/vec.h"

namespace Color {

inline float3 randomColor() {
  float3 c = make_float3(0);
  for (int i = 0; i < 3; ++i) {
	//random not supported in VC++
    c[i] = rand() / static_cast<float>(RAND_MAX);
  }
  return c;
}

// Returned color may not be close to avoid
inline float3 randomColor(const float3& avoid) {
  static const float DIST_THRESH = .1;
  float3 c = randomColor();
  while (length2(avoid - c) < DIST_THRESH) {
    c = randomColor();
  }
  return c;
}

// Returned color may not be close to avoid
inline float3 randomColor(int seed, const float3& avoid) {
  //srandom not supported in VC++
  srand(seed+1);
  return randomColor(avoid);
}

}

#endif
