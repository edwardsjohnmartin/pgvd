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

#include "../Dimension/dim.h"

//------------------------------------------------------------
//------------------------------------------------------------
// C++
//------------------------------------------------------------
//------------------------------------------------------------
#ifndef __OPENCL_VERSION__

#include <stdlib.h>
#include <assert.h>

//#ifdef __cplusplus
//#include <iostream>
//#endif // __cplusplus

#ifdef __OPEN_CL_SUPPORT__
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif
#endif

// typedef bool uchar;
#ifdef __OPEN_CL_SUPPORT__
typedef cl_uchar uchar;
#else
typedef unsigned char uchar;
#endif

#ifndef __cplusplus
typedef unsigned char bool;
#endif // __cplusplus

inline int convert_int(const float f) {
  return (int)(f+0.5);
}
inline bool any(const bool b) {
  return b;
}

#define __CONST__ const
#define __GLOBAL__
#define global
#define NAMESPACE_OCT_BEGIN namespace oct {
#define NAMESPACE_OCT_END }

#ifdef OCT2D
static __CONST__ int kSubAdded = 5;
#else
static __CONST__ int kSubAdded = 19;
#endif

#ifdef __cplusplus

#define M_PI 3.14159265359F

//template <typename T, typename P>
//void CompareAndExit(const T& a, const T& b, const P p,
//                    const std::string prefix) {
//  if (a != b) {
//    std::cout << prefix << " (" << p << "): " << a << " " << b << std::endl;
//    std::cout << "CompareAndExit exiting" << std::endl;
//    exit(0);
//  }
//}
//
//template <typename T>
//void CompareAndExit(const T& a, const T& b,
//                    const std::string prefix) {
//  if (a != b) {
//    std::cout << prefix << ": " << a << " " << b << std::endl;
//    std::cout << "CompareAndExit exiting" << std::endl;
//    exit(0);
//  }
//}
#endif // __cplusplus

#else
typedef short cl_short;
typedef int cl_int;
typedef float cl_float;
// typedef double cl_double;
typedef unsigned cl_uint;
#endif

//------------------------------------------------------------
//------------------------------------------------------------
// Generic
//------------------------------------------------------------
//------------------------------------------------------------

#define EPSILON 1e-6F

#ifndef nullptr
#define nullptr 0
#endif


#ifdef OCT2D
//------------------------------------------------------------
// 2D
//------------------------------------------------------------
// #define DIM 2
// #define Face Edge
// #include "./edge.h"
// // static __CONST__ int kNumSubdivided = 9;
// #define kNumSubdivided 9
// // The maximum number of octree vertices that can be created
// // in a subdivide operation.  3^D-2^D
// // static __CONST__ int kSubAdded = 5;
// static __CONST__ int kNumNewVertices = 5;
// static __CONST__ int kNumIncidentVertices = 8;
// static __CONST__ int kNumIncidentCells = 12; // 4*2 + 4
#else
//------------------------------------------------------------
// 3D
//------------------------------------------------------------
// #define DIM 3
// #define Face Triangle
// #include "./triangle.h"
// // static __CONST__ int kNumSubdivided = 27;
// #define kNumSubdivided 27
// // The maximum number of octree vertices that can be created
// // in a subdivide operation.  3^D-2^D
// // static __CONST__ int kSubAdded = 19;
// static __CONST__ int kNumNewVertices = 19;
// static __CONST__ int kNumIncidentVertices = 26;
// // static __CONST__ int kNumIncidentCells = 56; // 6*4 + 12*2 + 8
// #define kNumIncidentCells 56
#endif

// NAMESPACE_OCT_BEGIN

// Ordering is
//
//        ______________________
//       /          /          /|
//      /    6     /    7     / |
//     /__________/__________/  |
//    /          /          /| 7|
//   /    2     /    3     / |  |
//  /__________/__________/  |  |
// |          |           | 3| /|
// |          |           |  |/ |
// |    2     |     3     |  |  |
// |          |           | /| 5|
// |__________|___________|/ |  /
// |          |           | 1| /
// |          |           |  |/
// |    0     |     1     |  /
// |          |           | /
// |__________|___________|/
//
// Storage is by level.  Each cell stores an instance
// of T and the index of its children (0 if leaf).

//------------------------------------------------------------------------------
// struct Constants
//------------------------------------------------------------------------------

typedef int index_t;
typedef uchar level_t;
// typedef int level_t;

// Can't use variables (even static const) in initialization of static
// constants or else it won't compile in OpenCL.

#ifndef __OPENCL_VERSION__
static __CONST__ level_t kMaxLevel = 24;
#endif
// inline index_t Level2CellWidth(const level_t level) {
//   return kWidth >> level;
// }

// NAMESPACE_OCT_END

