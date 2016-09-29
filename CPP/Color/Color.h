#pragma once

#include "../../C/Vector/vec_n.h"
#include <ctime>
#include <cstdlib>
namespace Color {
  inline float_3 randomColor() {
    float_3 c = { 0.0, 0.0, 0.0 };
    int degree = rand()%360;
    int hueSection = degree/60;
    switch ( hueSection ) {
	  case 0:
		  c.s[0] = 1.0;
      c.s[1] = (degree/60.0);
      c.s[2] = 0.0;
		  break;
	  case 1:
		  c.s[0] = 1.0-(degree-60.0)/60.0;
      c.s[1] = 1.0;
      c.s[2] = 0.0;
		  break;
	  case 2:
		  c.s[0] = 0.0;
      c.s[1] = 1.0;
      c.s[2] = ((degree-120.0) / 60.0);
		  break;
	  case 3:
		  c.s[0] = 0.0;
      c.s[1] = 1.0 - (degree - 180.0) / 60.0;
      c.s[2] = 1.0;
		  break;
	  case 4:
		  c.s[0] = ((degree-240.0) / 60.0);
      c.s[1] = 0.0;
      c.s[2] = 1.0;
	    break;
	  case 5:
		  c.s[0] = 1.0;
      c.s[1] = 1.0*(degree/60.0);
      c.s[2] = 1.0 - (degree - 300.0) / 60.0;
		  break;
	  }
	  return c;
  }
}
