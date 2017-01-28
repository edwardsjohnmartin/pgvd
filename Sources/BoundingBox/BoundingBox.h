/*******************************************************
 ** Generalized Voronoi Diagram Project               **
 ** Copyright (c) 2015 John Martin Edwards            **
 ** Scientific Computing and Imaging Institute        **
 ** 72 S Central Campus Drive, Room 3750              **
 ** Salt Lake City, UT 84112                          **
 **                                                   **
 ** For information about this project contact        **
 ** John Edwards at                                   **
 **    edwardsjohnmartin@gmail.com                    **
 ** or visit                                          **
 **    sci.utah.edu/~jedwards/research/gvd/index.html **
 *******************************************************/
#pragma once

#ifdef OpenCL 
	#include "./Sources/Dimension/dim.h"
  #include "./Sources/Vector/vec.h"
  #include "./Sources/Quantize/Quantize.h"
#elif defined __cplusplus
extern "C" {
	#include "Dimension/dim.h"
}
  #include "Quantize/Quantize.h"
	#include "Vector/vec.h"
#else 
#error Only C++/OpenCL can compile BoundingBox.h!
#endif

typedef struct BoundingBox {
  bool initialized;
  floatn minimum;
  floatn maximum;
  float maxwidth;
} BoundingBox;

void do_stuff(const floatn* v);

inline float max_in_floatn(const floatn *a) {
	float result = a->s[0];
	for (int i = 1; i < DIM; ++i)	
		result = (a->s[i] > result) ? a->s[i] : result;
    return result;
}

inline int max_in_intn(const intn *a) {
  int result = a->s[0];
  for (int i = 1; i < DIM; ++i)
    result = (a->s[i] > result) ? a->s[i] : result;
  return result;
}

inline floatn BB_size(const BoundingBox *bb) {
	return bb->maximum - bb->minimum;
}

inline float BB_max_size(const BoundingBox *bb) {
	floatn size = BB_size(bb);
	return max_in_floatn(&size);
}

inline BoundingBox BB_initialize(const floatn* minimum, const floatn* maximum) {
    BoundingBox bb;
	bb.initialized = true;
	bb.minimum = *minimum;
	bb.maximum = *maximum;
    bb.maxwidth = BB_max_size(&bb);
    return bb;
}

inline int BB_max_quantized_size(const BoundingBox *bb, const int reslnWidth) {
  float bbSize = BB_max_size(bb);
  intn qMax = QuantizePoint(&bb->maximum, &bb->minimum, reslnWidth, bbSize);
  return max_in_intn(&qMax) + 1;
}

inline floatn BB_center(const BoundingBox *bb) {
	return (bb->minimum + bb->maximum) / 2.0;
}

inline intn BB_quantized_center(const BoundingBox *bb, const int reslnWidth) {
  /* There might be a better way to do this... */
  floatn fCenter = (bb->minimum + bb->maximum) / 2.0;
  float bbSize = BB_max_size(bb);
  return QuantizePoint(&fCenter, &bb->minimum, reslnWidth, bbSize);
}

inline bool BB_contains_point(const BoundingBox *bb, floatn *point, const float epsilon) {
	for (int i = 0; i < DIM; ++i)
		if ((point->s[i] <= bb->minimum.s[i] - epsilon) ||
			(point->s[i] >= bb->maximum.s[i] + epsilon))
			return false;
	return true;
}

// Returns the smallest square bounding box that contains
// bb and has identical origin.
inline BoundingBox BB_make_square(const BoundingBox *bb) {
	floatn size = BB_size(bb);
	float dwidth = max_in_floatn(&size);
	BoundingBox result = BB_initialize(&bb->minimum, &bb->minimum);
	result.maximum += dwidth;
    result.maxwidth = BB_max_size(&result);
    return result;
}

inline BoundingBox BB_make_centered_square(const BoundingBox *bb) {
	floatn size = BB_size(bb);
	float dwidth = max_in_floatn(&size);

	//The resulting box is square, so init both in and max with original min. Then add dwidth to max.
	BoundingBox result = BB_initialize(&bb->minimum, &bb->minimum);
	result.maximum += dwidth;

	for (int i = 0; i < DIM; ++i) {
		result.minimum.s[i] -= (dwidth - size.s[i]) / 2.0F;
		result.maximum.s[i] = result.minimum.s[i] + dwidth;
	}
	result.initialized = true;
    result.maxwidth = BB_max_size(&result);
    return result;
}

inline BoundingBox BB_scale(const BoundingBox *bb, const float f) {
	floatn size = BB_size(bb);
	size *= f;

	floatn newMax = bb->minimum + size;
	return BB_initialize(&bb->minimum, &newMax);
}

inline BoundingBox BB_scale_centered(BoundingBox *result, const BoundingBox *bb, const float f) {
	floatn s = BB_size(bb);
	s *= f;
	s /= 2.0;

	floatn c = BB_center(bb);

	floatn minp = c - s;
	floatn maxp = c + s;

	return BB_initialize(&minp, &maxp);
}

inline bool BB_is_square(const BoundingBox *bb) {
	floatn s = BB_size(bb);
	float a = s.x;
	for (int i = 1; i < DIM; ++i) {
		if (s.s[i] != a) return false;
	}
	return true;
}

#ifdef __cplusplus
#include <iostream>
inline std::ostream& operator<<(std::ostream& out, const BoundingBox& bb) {
  out << bb.minimum << " " << bb.maximum;
  return out;
}
#endif
