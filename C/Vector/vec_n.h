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

#define OCT2D
#ifdef OCT2D
  #ifdef __OPENCL_VERSION__
    #define CREATE_N_VEC(type) typedef union type##n {\
      type##2 v;\
      struct { type x, y; };\
      type s[DIM];\
    } type## n;
    CREATE_N_VEC(int)
    CREATE_N_VEC(float)
    CREATE_N_VEC(double)

  #else 
    #include "CL/cl.h"
    #define CREATE_N_VEC(type) typedef union type##n {\
      cl_##type##2 v;\
      struct { type x, y; };\
      type s[DIM];\
    } type## n;
    CREATE_N_VEC(int)
    CREATE_N_VEC(float)
    CREATE_N_VEC(double)
  #endif
#else
  #ifdef __OPENCL_VERSION__
    typedef int3 intn;
    typedef float3 floatn;
    typedef double3 doublen;
  #else 
    #include "CL/cl.h"
    typedef cl_int3 intn;
    typedef cl_float3 floatn;
    typedef cl_double3 doublen;
  #endif
#endif

#ifndef __OPENCL_VERSION__
    typedef cl_float2 floatnn;
    typedef cl_int2 int2;
    typedef cl_float2 float2;
    typedef cl_double2 double2;
    typedef cl_int3 int3;
    typedef cl_float3 float3;
    typedef cl_double3 double3;
#endif

/* C/C++ vector math */

/* Addition and Subtraction*/
inline intn* add_iviv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((intn*)result)->s[i] = ((intn*)a)->s[i] + ((intn*)b)->s[i];
  return (intn*)result;
}
inline floatn* add_fvfv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = ((floatn*)a)->s[i] + ((floatn*)b)->s[i];
  return (floatn*)result;
}
inline floatn* add_fvfv_by_val(void *result, const floatn a, const floatn b) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = a.s[i] + b.s[i];
  return (floatn*)result;
}
inline floatn* add_ffv(void *result, const float a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = a + ((floatn*)b)->s[i];
  return (floatn*)result;
}

inline floatn* subt_fvfv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = ((floatn*)a)->s[i] - ((floatn*)b)->s[i];
  return (floatn*)result;
}

/* Scalar multiplication and division*/
inline floatn* mult_fvf(void *result, const void *a, const float b) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = ((floatn*)a)->s[i] * b;
  return (floatn*)result;
}
inline floatn* div_fvf(void *result, const void *a, const float b) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = ((floatn*)a)->s[i] / b;
  return (floatn*)result;
}

/* Minimum/Maximum */
inline floatn* min_fvfv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = (((floatn*)a)->s[i] < ((floatn*)b)->s[i] ? ((floatn*)a)->s[i] : ((floatn*)b)->s[i]);
  return (floatn*)result;
}
inline floatn* max_fvfv(void *result, const void *a, const void *b) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = (((floatn*)a)->s[i] > ((floatn*)b)->s[i] ? ((floatn*)a)->s[i] : ((floatn*)b)->s[i]);
  return (floatn*)result;
}
inline void max_in_fv(float *result, const void *a) {
  *result = ((floatn*)a)->s[0];
  for (int i = 1; i < DIM; ++i)
    *result = (((floatn*)a)->s[i] > *result) ? ((floatn*)a)->s[i] : *result;
}
inline void min_in_fv(float *result, const void *a) {
  *result = ((floatn*)a)->s[0];
  for (int i = 0; i < DIM; ++i) 
    *result = (((floatn*)a)->s[i] < *result) ? ((floatn*)a)->s[i] : *result;
}

/*Deep copy*/
inline void copy_fvfv(void *result, const void *a) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = ((floatn*)a)->s[i];
}
inline void copy_fvf(void* result, const float a) {
  for (int i = 0; i < DIM; i++)
    ((floatn*)result)->s[i] = a;
}
