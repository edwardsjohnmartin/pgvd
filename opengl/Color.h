#ifndef __COLOR_H__
#define __COLOR_H__

#include "../opencl/vec.h"
#include <ctime>
#include <cstdlib>
namespace Color {

inline float3 randomColor() {
  float3 c = make_float3(0);
  int degree = rand()%360;
  int hueSection = degree/60;
  switch ( hueSection ) {
	case 0:
		c.x = 1.0;
    c.y = (degree/60.0);
    c.z = 0.0;
		break;
	case 1:
		c.x = 1.0-(degree-60.0)/60.0;
    c.y = 1.0;
    c.z = 0.0;
		break;
	case 2:
		c.x = 0.0;
    c.y = 1.0;
    c.z = ((degree-120) / 60.0);
		break;
	case 3:
		c.x = 0.0;
    c.y = 1.0 - (degree - 180.0) / 60.0;
    c.z = 1.0;
		break;
	case 4:
		c.x = ((degree-240) / 60.0);
    c.y = 0.0;
    c.z = 1.0;
	  break;
	case 5:
		c.x = 1.0;
    c.y = 1.0*(degree/60);
    c.z = 1.0 - (degree - 300.0) / 60.0;
		break;
	}
	return c;
}

// Returned color may not be close to avoid. Also avoids black.
inline float3 randomColor(int seed, const float3& avoid) {
  srand((seed+1)*20);
  return randomColor();
}

}

#endif
