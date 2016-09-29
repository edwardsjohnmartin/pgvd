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

#ifdef __OPENCL_VERSION__ 
  #include "./OpenCL/C/Vector/vec_n.h"
  #include "./OpenCL/C/Boolean/bool.h"
#else
  #include "../Vector/vec_n.h"
  #include "../Boolean/bool.h"
#endif

typedef struct BoundingBox {
  bool initialized;
  float_n minimum;
  float_n maximum;
} BoundingBox;

void BB_initialize(BoundingBox *bb, const float_n* minimum, const float_n* maximum);

void BB_center(const BoundingBox *bb, float_n *center);

void BB_size(const BoundingBox *bb, float_n *size);

void BB_max_size(const BoundingBox *bb, float *m);

bool BB_contains_point(const BoundingBox *bb, float_n *point, const float epsilon);

// Returns the smallest square bounding box that contains
// bb and has identical origin.
void BB_make_square(BoundingBox *result, const BoundingBox *bb);

void BB_make_centered_square(BoundingBox *result, const BoundingBox *bb);

void BB_scale(BoundingBox *result, const BoundingBox *bb, const float f);

void BB_scale_centered(BoundingBox *result, const BoundingBox *bb, const float f);

bool BB_is_square(const BoundingBox *bb);

