#pragma once

#include "../../C/Vector/vec_n.h"
#include <ctime>
#include <cstdlib>
namespace Color {
  inline cl_float3 randomColor() {
    cl_float3 c = { 0.0, 0.0, 0.0 };
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
      c.z = ((degree-120.0) / 60.0);
		  break;
	  case 3:
		  c.x = 0.0;
      c.y = 1.0 - (degree - 180.0) / 60.0;
      c.z = 1.0;
		  break;
	  case 4:
		  c.x = ((degree-240.0) / 60.0);
      c.y = 0.0;
      c.z = 1.0;
	    break;
	  case 5:
		  c.x = 1.0;
      c.y = 1.0*(degree/60.0);
      c.z = 1.0 - (degree - 300.0) / 60.0;
		  break;
	  }
	  return c;
  }
}
