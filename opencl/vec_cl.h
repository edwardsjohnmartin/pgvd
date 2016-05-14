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

#ifndef __VEC_CL_H__
#define __VEC_CL_H__
#define OCT2D;
#ifdef OCT2D
typedef int2 intn;
typedef float2 floatn;
#ifndef __OPENCL_VERSION__
typedef double2 doublen;
#endif
#else
typedef int3 intn;
typedef float3 floatn;
typedef double3 doublen;
#endif

#endif
