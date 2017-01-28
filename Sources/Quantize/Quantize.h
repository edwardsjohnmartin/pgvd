#pragma once

#include "Vector/vec.h"

#define IMIN(a,b) ((int)(((a)<(b))?(a):(b)))
inline intn QuantizePoint(const floatn *p, const floatn *minimum, const int reslnWidth, const float bbWidth) {
	int effectiveWidth = reslnWidth;
	floatn d_ = (((*p - *minimum) / bbWidth) * effectiveWidth);
	return make_intn(IMIN(d_.x, reslnWidth-1), IMIN(d_.y, reslnWidth-1));
}

inline floatn UnquantizePoint(const intn *p, const floatn *minimum, const int reslnWidth, const float bbWidth) {
  int effectiveWidth = reslnWidth;
  floatn q = make_floatn(p->x, p->y);
  q = (((q) / effectiveWidth) * bbWidth) + *minimum;
  return q;
}
