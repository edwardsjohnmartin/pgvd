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
#ifndef __OPENCL_VERSION__
#error vec_cl.h can only be compiled with OpenCL!
#endif

#pragma once

#ifdef OCT2D
typedef int2 intn;
typedef float2 floatn;
typedef double2 doublen;
#else
typedef int3 intn;
typedef float3 floatn;
typedef double3 doublen;
#endif

#define __CONST__ constant
#define __GLOBAL__ __global
#define NAMESPACE_OCT_BEGIN
#define NAMESPACE_OCT_END

#ifdef OCT2D
#define convert_intn convert_int2
#define convert_floatn convert_float2
#define convert_doublen convert_double2
#else
#define convert_intn convert_int3
#define convert_floatn convert_float3
#define convert_doublen convert_double3
#endif

 // Construction
#define make_bool2 (bool2)
#define make_int2 (int2)
#define make_float2 (float2)
#define make_double2 (double2)

#define make_bool3 (bool3)
#define make_int3 (int3)
#define make_float3 (float3)
#define make_double3 (double3)

#define make_bool4 (bool4)
#define make_int4 (int4)
#define make_float4 (float4)
#define make_double4 (double4)

#ifdef OCT2D
#define make_booln (bool2)
#define make_intn (int2)
#define make_floatn (float2)
#define make_doublen (double2)
#else
#define make_booln (bool3)
#define make_intn (int3)
#define make_floatn (float3)
#define make_doublen (double3)
#endif

// Makes a uniform vector, that is, one with identical values
// in each dimension.
// make_uni_int3, make_uni_float2, etc.
#define make_uni_bool2(a) (bool2)(a, a)
#define make_uni_int2(a) (int2)(a, a)
#define make_uni_float2(a) (float2)(a, a)
#define make_uni_double2(a) (double2)(a, a)

#define make_uni_bool3(a) (bool3)(a, a, a)
#define make_uni_int3(a) (int3)(a, a, a)
#define make_uni_float3(a) (float3)(a, a, a)
#define make_uni_double3(a) (double3)(a, a, a)

#define make_uni_bool4(a) (bool4)(a, a, a, a)
#define make_uni_int4(a) (int4)(a, a, a, a)
#define make_uni_float4(a) (float4)(a, a, a, a)
#define make_uni_double4(a) (double4)(a, a, a, a)

#ifdef OCT2D
#define make_uni_booln(a) (bool2)(a, a)
#define make_uni_intn(a) (int2)(a, a)
#define make_uni_floatn(a) (float2)(a, a)
#define make_uni_doublen(a) (double2)(a, a)
#else
#define make_uni_booln(a) (bool3)(a, a, a)
#define make_uni_intn(a) (int3)(a, a, a)
#define make_uni_floatn(a) (float3)(a, a, a)
#define make_uni_doublen(a) (double3)(a, a, a)
#endif
