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
 #include "./OpenCL/C/Dimension/dim.h"
#else
 #include "../../C/Dimension/dim.h"
#endif

#define CREATE_VEC(type, dim) typedef struct type ## _ ## dim{\
	type s[dim];\
} type ## _ ## dim;

#define CREATE_ALL(type) \
	CREATE_VEC(type, 2) \
	CREATE_VEC(type, 4) \
	typedef type ## _ ## 4 type ## _ ## 3;

#define CREATE_VEC_N(type) typedef struct type ## _n{\
	type s[DIM];\
} type ## _n;

CREATE_ALL(int)
CREATE_ALL(float)
CREATE_ALL(double)

CREATE_VEC_N(int)
CREATE_VEC_N(float)
CREATE_VEC_N(double)

#define X_(v) ((v).s[0])
#define Y_(v) ((v).s[1])
#define Z_(v) ((v).s[2])
#define W_(v) ((v).s[3])

/* C/C++ vector math */

/* Addition and Subtraction*/
inline int_n* add_iviv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((int_n*)result)->s[i] = ((int_n*)a)->s[i] + ((int_n*)b)->s[i];
  return (int_n*)result;
}
inline float_n* add_fvfv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = ((float_n*)a)->s[i] + ((float_n*)b)->s[i];
  return (float_n*)result;
}
inline float_n* add_fvfv_by_val(void *result, const float_n a, const float_n b) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = a.s[i] + b.s[i];
  return (float_n*)result;
}
inline float_n* add_ffv(void *result, const float a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = a + ((float_n*)b)->s[i];
  return (float_n*)result;
}

inline float_n* subt_fvfv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = ((float_n*)a)->s[i] - ((float_n*)b)->s[i];
  return (float_n*)result;
}

/* Scalar multiplication and division*/
inline float_n* mult_fvf(void *result, const void *a, const float b) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = ((float_n*)a)->s[i] * b;
  return (float_n*)result;
}
inline float_n* div_fvf(void *result, const void *a, const float b) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = ((float_n*)a)->s[i] / b;
  return (float_n*)result;
}

/* Minimum/Maximum */
inline float_n* min_fvfv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = (((float_n*)a)->s[i] < ((float_n*)b)->s[i] ? ((float_n*)a)->s[i] : ((float_n*)b)->s[i]);
  return (float_n*)result;
}
inline float_n* max_fvfv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = (((float_n*)a)->s[i] > ((float_n*)b)->s[i] ? ((float_n*)a)->s[i] : ((float_n*)b)->s[i]);
  return (float_n*)result;
}
inline void max_in_fv(float *result, const void *a) {
  *result = ((float_n*)a)->s[0];
  for (int i = 1; i < DIM; ++i)
    *result = (((float_n*)a)->s[i] > *result) ? ((float_n*)a)->s[i] : *result;
}
inline void min_in_fv(float *result, const void *a) {
  *result = ((float_n*)a)->s[0];
  for (int i = 0; i < DIM; ++i) 
    *result = (((float_n*)a)->s[i] < *result) ? ((float_n*)a)->s[i] : *result;
}

/*Deep copy*/
inline void copy_fvfv(void *result, const void *a) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = ((float_n*)a)->s[i];
}
inline void copy_fvf(void* result, const float a) {
  for (int i = 0; i < DIM; i++)
    ((float_n*)result)->s[i] = a;
}
