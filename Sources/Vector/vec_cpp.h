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

#ifndef __VEC_CPP_H__
#define __VEC_CPP_H__

#include <stdexcept>
#include <cmath>
#include <iostream>


//------------------------------------------------------------
// Use OpenCL's types
//------------------------------------------------------------
#define CL_VEC
#include "cl.hpp"

#ifndef __APPLE__
typedef cl_uchar2 bool2;
typedef cl_int2 int2;
typedef cl_float2 float2;
typedef cl_double2 double2;
typedef cl_uchar3 bool3;
typedef cl_int3 int3;
typedef cl_float3 float3;
typedef cl_double3 double3;
typedef cl_uchar4 bool4;
typedef cl_int4 int4;
typedef cl_float4 float4;
typedef cl_double4 double4;
#else
//------------------------------------------------------------
// Define our own type
//------------------------------------------------------------
template <class NumType, int NumDims>
union MyVec2 {
  struct { NumType s[NumDims]; };
  struct { NumType x, y; };
};

template <class NumType, int NumDims>
union MyVec3 {
  struct { NumType s[NumDims]; };
  struct { NumType x, y, z, w; };
};

typedef MyVec2<unsigned char, 2> bool2;
typedef MyVec2<int, 2> int2;
typedef MyVec2<float, 2> float2;
typedef MyVec2<double, 2> double2;

typedef MyVec3<int, 3> int3;
typedef MyVec3<float, 3> float3;
typedef MyVec3<double, 3> double3;
typedef MyVec3<unsigned char, 3> bool3;

typedef MyVec3<unsigned char, 4> bool4;
typedef MyVec3<int, 4> int4;
typedef MyVec3<float, 4> float4;
typedef MyVec3<double, 4> double4;
#endif

//------------------------------------------------------------
// Define typen
//------------------------------------------------------------
#ifdef OCT2D
typedef bool2 booln;
typedef int2 intn;
typedef float2 floatn;
typedef double2 doublen;
#else
typedef bool3 booln;
typedef int3 intn;
typedef float3 floatn;
typedef double3 doublen;
#endif

//------------------------------------------------------------
// PointAndLabel struct
//------------------------------------------------------------
struct PointAndLabel {
  PointAndLabel(const intn& p_, const cl_int l_) : p(p_), l(l_) {}
  intn p;
  cl_int l;
};

//------------------------------------------------------------
// Functions
//------------------------------------------------------------

// Construction

// make_int3, make_float2, etc.
#define MAKE_VECN(typen,type)                                         \
  inline typen make_##typen(type a=0, type b=0, type c=0, type d=0) {   \
    type arr[] = { a, b, c, d };                                        \
    return *(typen*)(arr);                                              \
    }
MAKE_VECN(bool2,cl_bool)
MAKE_VECN(int2,cl_int)
MAKE_VECN(float2,cl_float)
MAKE_VECN(double2, cl_double)
MAKE_VECN(bool3, cl_bool)
MAKE_VECN(int3, cl_int)
MAKE_VECN(float3, cl_float)
MAKE_VECN(double3, cl_double)
MAKE_VECN(bool4, cl_bool)
MAKE_VECN(int4, cl_int)
MAKE_VECN(float4, cl_float)
MAKE_VECN(double4, cl_double)

// #ifdef OCT2D
MAKE_VECN(booln, cl_bool)
MAKE_VECN(intn, cl_int)
MAKE_VECN(floatn, cl_float)
MAKE_VECN(doublen, cl_double)
// #else
// MAKE_VECN(booln,cl_bool)
// MAKE_VECN(intn,cl_int)
// MAKE_VECN(floatn,cl_float)
// MAKE_VECN(doublen,cl_double)
// #endif

// Makes a uniform vector, that is, one with identical values
// in each dimension.
// make_uni_int3, make_uni_float2, etc.
#define MAKE_UNI_VECN(typen,type)                                           \
  inline typen make_uni_##typen(type a) {                               \
    type arr[] = { a, a, a, a };                                        \
    return *(typen*)(arr);                                              \
    }
MAKE_UNI_VECN(bool2, cl_bool)
MAKE_UNI_VECN(int2, cl_int)
MAKE_UNI_VECN(float2, cl_float)
MAKE_UNI_VECN(double2, cl_double)
MAKE_UNI_VECN(bool3, cl_bool)
MAKE_UNI_VECN(int3, cl_int)
MAKE_UNI_VECN(float3, cl_float)
MAKE_UNI_VECN(double3, cl_double)
MAKE_UNI_VECN(bool4, cl_bool)
MAKE_UNI_VECN(int4, cl_int)
MAKE_UNI_VECN(float4, cl_float)
MAKE_UNI_VECN(double4, cl_double)

// #ifdef OCT2D
MAKE_UNI_VECN(booln, cl_bool)
MAKE_UNI_VECN(intn, cl_int)
MAKE_UNI_VECN(floatn, cl_float)
MAKE_UNI_VECN(doublen, cl_double)
// #else
// MAKE_UNI_VECN(booln,cl_bool)
// MAKE_UNI_VECN(intn,cl_int)
// MAKE_UNI_VECN(floatn,cl_float)
// MAKE_UNI_VECN(doublen,cl_double)
// #endif

// make_int3(v, 10) // where v is of type int2
#define MAKE_VECN_(typen,typen_,type,n)                         \
  inline typen make_##typen(const typen_& v_, type a=0) {       \
    type arr[] = { a, a, a, a };                                \
    for (cl_int i = 0; i < n-1; ++i) {                             \
      arr[i] = v_.s[i];                                         \
    }                                                           \
    return *(typen*)(arr);                                      \
  }
MAKE_VECN_(bool3,bool2, cl_bool,3)
MAKE_VECN_(int3,int2, cl_int,3)
MAKE_VECN_(float3,float2, cl_float,3)
MAKE_VECN_(double3,double2, cl_double,3)
MAKE_VECN_(bool4,bool3, cl_bool,4)
MAKE_VECN_(int4,int3, cl_int,4)
MAKE_VECN_(float4,float3, cl_float,4)
MAKE_VECN_(double4,double3, cl_double,4)
// //------------------------------------------------------------
// // cast
// #define VEC_2_POINTER(typen,type)                           \
//   inline operator const type*(const typen& v) {      \
//     return &v.s[0];                                     \
//   }
// VEC_2_POINTER(int2,cl_int)
// VEC_2_POINTER(float2,cl_float)
// VEC_2_POINTER(double2,cl_double)
// VEC_2_POINTER(int3,cl_int)
// VEC_2_POINTER(float3,cl_float)
// VEC_2_POINTER(double3,cl_double)

//------------------------------------------------------------
// output operators
#define OUT_VEC(typen,n)                                        \
  inline std::ostream& operator<<(std::ostream& out, const typen& v) { \
    for (cl_int i = 0; i < n; i++) {                               \
      out << v.s[i];                                            \
      if (i < n-1) out << " ";                                  \
    }                                                           \
    return out;                                                 \
  }
OUT_VEC(bool2,2)
OUT_VEC(int2,2)
OUT_VEC(float2,2)
OUT_VEC(double2,2)
#ifndef CL_VEC
OUT_VEC(bool3,3)
OUT_VEC(int3,3)
OUT_VEC(float3,3)
OUT_VEC(double3,3)
#endif
OUT_VEC(bool4,4)
OUT_VEC(int4,4)
OUT_VEC(float4,4)
OUT_VEC(double4,4)
#define IN_VEC(typen,n)                                         \
  inline std::istream& operator>>(std::istream& in, typen& v) {        \
    for (cl_int i = 0; i < n; i++) {                               \
      in >> v.s[i];                                             \
    }                                                           \
    return in;                                                  \
  }
IN_VEC(bool2,2)
IN_VEC(int2,2)
IN_VEC(float2,2)
IN_VEC(double2,2)
#ifndef CL_VEC
IN_VEC(bool3,3)
IN_VEC(int3,3)
IN_VEC(float3,3)
IN_VEC(double3,3)
#endif
IN_VEC(bool4,4)
IN_VEC(int4,4)
IN_VEC(float4,4)
IN_VEC(double4,4)

//------------------------------------------------------------
// min/max
#define VEC_VEC_MIN(typen,n)                                    \
  inline typen vec_min(const typen& a, const typen& b) {                \
    typen result = make_##typen(); \
    for (cl_int i = 0; i < n; i++) {                               \
      result.s[i] = (a.s[i] < b.s[i] ? a.s[i] : b.s[i]);        \
    }                                                           \
    return result;                                              \
  }
VEC_VEC_MIN(int2,2)
VEC_VEC_MIN(float2,2)
VEC_VEC_MIN(double2,2)
VEC_VEC_MIN(int3,3)
VEC_VEC_MIN(float3,3)
VEC_VEC_MIN(double3,3)
#define VEC_VEC_MAX(typen,n)                                    \
  inline typen vec_max(const typen& a, const typen& b) {            \
    typen result = make_##typen();                                    \
    for (cl_int i = 0; i < n; i++) {                               \
      result.s[i] = (a.s[i] > b.s[i] ? a.s[i] : b.s[i]);        \
    }                                                           \
    return result;                                              \
  }
VEC_VEC_MAX(int2,2)
VEC_VEC_MAX(float2,2)
VEC_VEC_MAX(double2,2)
VEC_VEC_MAX(int3,3)
VEC_VEC_MAX(float3,3)
VEC_VEC_MAX(double3,3)

//------------------------------------------------------------
// Binary operators
// All binary operators work component-wise on vectors
//------------------------------------------------------------

#define VEC_OP_SCALAR(typen,type,n,op)                          \
  inline typen operator op (const typen v, const type a) {      \
  typen result = make_##typen();                                \
    for (cl_int i = 0; i < n; i++) result.s[i] = v.s[i] op a;      \
    return result;                                              \
  }
#define VEC_OP_VEC(typen,n,op)                                  \
  inline typen operator op (const typen u, const typen v) {     \
    typen result = make_##typen();                                               \
    for (cl_int i = 0; i < n; i++) result.s[i] = u.s[i] op v.s[i]; \
    return result;                                              \
  }
#define VEC_OP_ASSIGN_SCALAR(typen,type,n,op)           \
  inline typen& operator op (typen& v, const type a) {  \
    for (cl_int i = 0; i < n; i++) v.s[i] op a;            \
    return v;                                           \
  }
#define VEC_OP_ASSIGN_VEC(typen,n,op)                   \
  inline typen& operator op (typen& u, const typen v) { \
    for (cl_int i = 0; i < n; i++) u.s[i] op v.s[i];       \
    return u;                                           \
  }
// v * a
VEC_OP_SCALAR(int2, cl_int, 2, *)
VEC_OP_SCALAR(float2, cl_float, 2, *)
VEC_OP_SCALAR(double2, cl_double, 2, *)
VEC_OP_SCALAR(int3, cl_int, 3, *)
VEC_OP_SCALAR(float3, cl_float, 3, *)
VEC_OP_SCALAR(double3, cl_double, 3, *)
// v *= a
VEC_OP_ASSIGN_SCALAR(int2, cl_int, 2, *=)
VEC_OP_ASSIGN_SCALAR(float2, cl_float, 2, *=)
VEC_OP_ASSIGN_SCALAR(double2, cl_double, 2, *=)
VEC_OP_ASSIGN_SCALAR(int3, cl_int, 3, *=)
VEC_OP_ASSIGN_SCALAR(float3, cl_float, 3, *=)
VEC_OP_ASSIGN_SCALAR(double3, cl_double, 3, *=)
// u * v
VEC_OP_VEC(int2, 2, *)
VEC_OP_VEC(float2, 2, *)
VEC_OP_VEC(double2, 2, *)
VEC_OP_VEC(int3, 3, *)
VEC_OP_VEC(float3, 3, *)
VEC_OP_VEC(double3, 3, *)
// v / a
VEC_OP_SCALAR(int2, cl_int, 2, /)
VEC_OP_SCALAR(float2, cl_float, 2, /)
VEC_OP_SCALAR(double2, cl_double, 2, /)
VEC_OP_SCALAR(int3, cl_int, 3, /)
VEC_OP_SCALAR(float3, cl_float, 3, /)
VEC_OP_SCALAR(double3, cl_double, 3, /)
// v /= a
VEC_OP_ASSIGN_SCALAR(int2, cl_int, 2, /=)
VEC_OP_ASSIGN_SCALAR(float2, cl_float, 2, /=)
VEC_OP_ASSIGN_SCALAR(double2, cl_double, 2, /=)
VEC_OP_ASSIGN_SCALAR(int3, cl_int, 3, /=)
VEC_OP_ASSIGN_SCALAR(float3, cl_float, 3, /=)
VEC_OP_ASSIGN_SCALAR(double3, cl_double, 3, /=)
// v + a
VEC_OP_SCALAR(double2, cl_double, 2, +)
VEC_OP_SCALAR(float2, cl_float, 2, +)
VEC_OP_SCALAR(int2, cl_int, 2, +)
VEC_OP_SCALAR(double3, cl_double, 3, +)
VEC_OP_SCALAR(float3, cl_float, 3, +)
VEC_OP_SCALAR(int3, cl_int, 3, +)
// v += a
VEC_OP_ASSIGN_SCALAR(int2, cl_int, 2, +=)
VEC_OP_ASSIGN_SCALAR(float2, cl_float, 2, +=)
VEC_OP_ASSIGN_SCALAR(double2, cl_double, 2, +=)
VEC_OP_ASSIGN_SCALAR(int3, cl_int, 3, +=)
VEC_OP_ASSIGN_SCALAR(float3, cl_float, 3, +=)
VEC_OP_ASSIGN_SCALAR(double3, cl_double, 3, +=)
// u + v
VEC_OP_VEC(double2, 2, +)
VEC_OP_VEC(float2, 2, +)
VEC_OP_VEC(int2, 2, +)
VEC_OP_VEC(double3, 3, +)
VEC_OP_VEC(float3, 3, +)
VEC_OP_VEC(int3, 3, +)
// u += v
VEC_OP_ASSIGN_VEC(double2, 2, +=)
VEC_OP_ASSIGN_VEC(float2, 2, +=)
VEC_OP_ASSIGN_VEC(int2, 2, +=)
VEC_OP_ASSIGN_VEC(double3, 3, +=)
VEC_OP_ASSIGN_VEC(float3, 3, +=)
VEC_OP_ASSIGN_VEC(int3, 3, +=)
// v - a
VEC_OP_SCALAR(double2, cl_double, 2, -)
VEC_OP_SCALAR(float2, cl_float, 2, -)
VEC_OP_SCALAR(int2, cl_int, 2, -)
VEC_OP_SCALAR(double3, cl_double, 3, -)
VEC_OP_SCALAR(float3, cl_float, 3, -)
VEC_OP_SCALAR(int3, cl_int, 3, -)
// u - v
VEC_OP_VEC(double2, 2, -)
VEC_OP_VEC(float2, 2, -)
VEC_OP_VEC(int2, 2, -)
VEC_OP_VEC(double3, 3, -)
VEC_OP_VEC(float3, 3, -)
VEC_OP_VEC(int3, 3, -)
// - (negate)
#define VEC_NEGATE(typen,n)                             \
  inline typen operator-(const typen v) {               \
    typen result = make_##typen();                                       \
    for (cl_int i = 0; i < n; i++) result.s[i] = -v.s[i];  \
    return result;                                      \
  }
VEC_NEGATE(double2, 2)
VEC_NEGATE(float2, 2)
VEC_NEGATE(int2, 2)
VEC_NEGATE(double3, 3)
VEC_NEGATE(float3, 3)
VEC_NEGATE(int3, 3)

//------------------------------------------------------------
// equality
#define VEC_EQUALS(typen,n)                                     \
  inline cl_bool operator==(const typen& a, const typen& b) {      \
    cl_bool eq = true;                                             \
    for (cl_int i = 0; i < n && (eq = (a.s[i] == b.s[i])); ++i);  \
    return eq;                                                  \
  }
VEC_EQUALS(int3,3)
VEC_EQUALS(float3,3)
VEC_EQUALS(double3,3)
VEC_EQUALS(int2,2)
VEC_EQUALS(float2,2)
VEC_EQUALS(double2,2)
#define VEC_NOT_EQUALS(typen,n)                                 \
  inline cl_bool operator!=(const typen& a, const typen& b) {      \
    return !(a == b);                                           \
  }
VEC_NOT_EQUALS(int3,3)
VEC_NOT_EQUALS(float3,3)
VEC_NOT_EQUALS(double3,3)
VEC_NOT_EQUALS(int2,2)
VEC_NOT_EQUALS(float2,2)
VEC_NOT_EQUALS(double2,2)
// <
#define VEC_LESS_SCALAR(typen,type,n)                           \
  inline cl_bool operator<(const typen& v, const type& a) {        \
    cl_bool lt = true;                                             \
    for (cl_int i = 0; i < n && (lt = (v.s[i] < a)); i++);         \
    return lt;                                                  \
  }
VEC_LESS_SCALAR(int3,cl_int,3)
VEC_LESS_SCALAR(float3,cl_float,3)
VEC_LESS_SCALAR(double3,cl_double,3)
VEC_LESS_SCALAR(int2,cl_int,2)
VEC_LESS_SCALAR(float2,cl_float,2)
VEC_LESS_SCALAR(double2,cl_double,2)
#define VEC_LESS_VEC(typen,n)                                   \
  inline cl_bool operator<(const typen& u, const typen& v) {       \
    for (cl_int i = 0; i < n; ++i) {                               \
      if (u.s[i] < v.s[i]) return true;                         \
      if (u.s[i] > v.s[i]) return false;                        \
    }                                                           \
    return false;                                               \
  }
VEC_LESS_VEC(int3,3)
VEC_LESS_VEC(float3,3)
VEC_LESS_VEC(double3,3)
VEC_LESS_VEC(int2,2)
VEC_LESS_VEC(float2,2)
VEC_LESS_VEC(double2,2)

//------------------------------------------------------------
// Type conversions
#define CONVERT_VEC3(from,to,type)                              \
  inline to convert_##to(const from v) {                        \
    return make_##to((type)v.s[0], (type)v.s[1], (type)v.s[2]); \
  }
#define CONVERT_VEC2(from,to,type)                      \
  inline to convert_##to(const from v) {                \
    return make_##to((type)v.s[0], (type)v.s[1]);       \
  }

#ifdef OCT2D
#define CONVERT_VECN(from,to,type)                      \
  inline to convert_##to(const from v) {                \
    return make_##to((type)v.s[0], (type)v.s[1]);       \
  }
#else
#define CONVERT_VECN(from,to,type)                              \
  inline to convert_##to(const from v) {                        \
    return make_##to((type)v.s[0], (type)v.s[1], (type)v.s[2]); \
  }
#endif

CONVERT_VEC3(float3,int3,cl_int)
CONVERT_VEC3(double3,int3,cl_int)
CONVERT_VEC3(int3,float3,cl_float)
CONVERT_VEC3(double3,float3,cl_float)
CONVERT_VEC3(int3,double3,cl_double)
CONVERT_VEC3(float3,double3,cl_double)

CONVERT_VEC2(float2,int2,cl_int)
CONVERT_VEC2(double2,int2,cl_int)
CONVERT_VEC2(int2,float2,cl_float)
CONVERT_VEC2(double2,float2,cl_float)
CONVERT_VEC2(int2,double2,cl_double)
CONVERT_VEC2(float2,double2,cl_double)

CONVERT_VECN(floatn,intn,cl_int)
CONVERT_VECN(doublen,intn,cl_int)
CONVERT_VECN(intn,floatn,cl_float)
CONVERT_VECN(doublen,floatn,cl_float)
CONVERT_VECN(intn,doublen,cl_double)
CONVERT_VECN(floatn,doublen,cl_double)

//------------------------------------------------------------
// L1norm
inline cl_int L1norm(const int2 a) {
  cl_int result = 0;
  for (cl_int i = 0; i < 2; i++) result += a.s[i];
  return result;
}
inline cl_int L1norm(const int3 a) {
  cl_int result = 0;
  for (cl_int i = 0; i < 3; i++) result += a.s[i];
  return result;
}
//------------------------------------------------------------
// dot
inline cl_double dot(const double3 a, const double3 b) {
  cl_double result = 0;
  for (cl_int i = 0; i < 3; i++) result += a.s[i]*b.s[i];
  return result;
}
inline cl_float dot(const float3 a, const float3 b) {
  cl_float result = 0;
  for (cl_int i = 0; i < 3; i++) result += a.s[i]*b.s[i];
  return result;
}
inline cl_double dot(const double2 a, const double2 b) {
  cl_double result = 0;
  for (cl_int i = 0; i < 2; i++) result += a.s[i]*b.s[i];
  return result;
}
inline cl_float dot(const float2 a, const float2 b) {
  cl_float result = 0;
  for (cl_int i = 0; i < 2; i++) result += a.s[i]*b.s[i];
  return result;
}
//------------------------------------------------------------
// length2
inline cl_double length2(const double3 v) {
  return dot(v, v);
}
inline cl_float length2(const float3 v) {
  return dot(v, v);
}
inline cl_int length2(const int3 v) {
  throw std::logic_error("can't get norm2 of integer type");
}
inline cl_double length2(const double2 v) {
  return dot(v, v);
}
inline cl_float length2(const float2 v) {
  return dot(v, v);
}
inline cl_int length2(const int2 v) {
  throw std::logic_error("can't get norm2 of integer type");
}
//------------------------------------------------------------
// length
#define fast_length length
inline cl_double length(const double3 v) {
  return sqrt(length2(v));
}
inline cl_float length(const float3 v) {
  return sqrt(length2(v));
}
inline cl_int length(const int3 v) {
  float3 vf = make_float3();
  for (cl_int i = 0; i < 3; ++i) {
    vf.s[i] = (float)v.s[i];
  }
  cl_float n = length(vf);
  return (cl_int)(n+0.5);
}
inline cl_double length(const double2 v) {
  return sqrt(length2(v));
}
inline cl_float length(const float2 v) {
  return sqrt(length2(v));
}
inline cl_int length(const int2 v) {
  float2 vf = make_float2();
  for (cl_int i = 0; i < 2; ++i) {
    vf.s[i] = v.s[i];
  }
  cl_float n = length(vf);
  return (cl_int)(n+0.5);
}
//------------------------------------------------------------
// normalize
#define fast_normalize normalize
inline double3 normalize(const double3 v) {
  return v / length(v);
}
inline float3 normalize(const float3 v) {
  return v / length(v);
}
inline int3 normalize(const int3 v) {
  return v / length(v);
}
inline double2 normalize(const double2 v) {
  return v / length(v);
}
inline float2 normalize(const float2 v) {
  return v / length(v);
}
inline int2 normalize(const int2 v) {
  return v / length(v);
}
inline double3 unit(const double3 v) {
  return v / length(v);
}
inline float3 unit(const float3 v) {
  return v / length(v);
}
inline int3 unit(const int3 v) {
  return v / length(v);
}
inline double2 unit(const double2 v) {
  return v / length(v);
}
inline float2 unit(const float2 v) {
  return v / length(v);
}
inline int2 unit(const int2 v) {
  return v / length(v);
}
//------------------------------------------------------------
// cross
#define VEC_CROSS_VEC(type3)                            \
  inline type3 cross(const type3 a, const type3 b) {    \
    type3 result = make_##type3();                      \
    result.s[0] = a.s[1]*b.s[2]-a.s[2]*b.s[1];          \
    result.s[1] = a.s[2]*b.s[0]-a.s[0]*b.s[2];          \
    result.s[2] = a.s[0]*b.s[1]-a.s[1]*b.s[0];          \
    return result;                                      \
  }
VEC_CROSS_VEC(int3)
VEC_CROSS_VEC(float3)
VEC_CROSS_VEC(double3)

#endif
