#ifndef __OPENCL_VERSION__
  #include "../BigUnsigned/BigUnsigned.h"
  #include "../Line/Line.h"
  #include "../Vector/vec_n.h"
  #include "CellResolution.h"
#else
  #include "./OpenCL/C/BigUnsigned/BigUnsigned.h"
  #include "./OpenCL/C/Line/Line.h"
  #include "./OpenCL/C/Vector/vec_n.h"
  #include "./OpenCL/CellResolution/CellResolution.h"
#endif

/* Ambiguous cells code */
unsigned char computeOutCode(floatn point, floatn min, floatn max) {
  unsigned char mask = 0;
  if (point.x < min.x)
    mask |= 1;
  else if (point.x > max.x)
    mask |= 1 << 1;
  if (point.y < min.y)
    mask |= 1 << 2;
  else if (point.y > max.y)
    mask |= 1 << 3;
  #if DIM == 3
    if (point.z < min.z)
      mask |= 1 << 4;
    else if (point.z > max.z)
      mask |= 1 << 5;
  #endif
  return mask;
}

void sub_v2v2(cl_double2 *a, cl_double2 b, cl_double2 c) {
  a->x = b.x - c.x;
  a->y = b.y - c.y;
}

void dot_v2v2(double *dot, cl_double2 a, cl_double2 b) {
  *dot = (a.x*b.x) + (a.y*b.y);
}

void point_on_vn(cl_double2 *result, cl_double2 point, cl_double2 ray, double t) {
  result->x = point.x + ray.x * t;
  result->y = point.y + ray.y * t;
  #if DIM == 3
    result->z = point.z + ray.z * t;
  #endif
}

bool v3_on_aasquare(cl_float3 point, cl_float3 min, cl_float3 normal, float width) {
  #define CHECK_FACE(a, b, c) \
  if (normal.a != 0.0) { return ((point.b > min.b && point.b < min.b + width) && (point.c > min.c && point.c < min.c + width)); }
  CHECK_FACE(x, y, z);
  CHECK_FACE(y, z, x);
  CHECK_FACE(z, x, y);
  #undef CHECK_FACE
  return false;
}

bool v2_on_aaedge(cl_double2 point, cl_double2 min, cl_double2 normal, float width) {
  #define CHECK_EDGE(a, b) \
  if (normal.a != 0.0) { return (point.b > min.b && point.b < min.b + width); }
  CHECK_EDGE(x, y);
  CHECK_EDGE(y, x);
  #undef CHECK_EDGE
  return false;
}

//Line-box intersection tests
int doCohenSutherlandTest(floatn point1, floatn point2, floatn min, floatn max) {
  unsigned char outcode1 = computeOutCode(point1, min, max);
  unsigned char outcode2 = computeOutCode(point2, min, max);
  /* one of the two points is in the center*/
  if (outcode1 == 0 || outcode2 == 0)
    return 1;
  /* Both points have a side in common */
  else if (outcode1 & outcode2)
    return -1;
  /* The points are accross the cube. */
  else if ((outcode1 == 1 && outcode2 == 2) || (outcode2 == 1 && outcode1 == 2)) {
    return 1;
  }
  else if ((outcode1 == 8 && outcode2 == 4) || (outcode2 == 8 && outcode1 == 4)) {
    return 1;
  }
  else if ((outcode1 == 32 && outcode2 == 16) || (outcode2 == 32 && outcode1 == 16)) {
    return 1;
  }
  /* This is an ambiguous case */
  return 0;
}
bool doLineBoxTest(const floatn *point1, const floatn *point2, const floatn *minimum, const floatn *maximum) {
#if DIM == 3
  cl_double3 dMinimum = { minimum->x, minimum->y, minimum->z };
  cl_double3 newPoint = {};

#else
  cl_double2 dMinimum = { minimum->x, minimum->y };
  cl_double2 dMaximum = { maximum->x, maximum->y };
  cl_double2 newPoint = {0.0,0.0};
  cl_double2 dPoint1 = { point1->x, point1->y };
  cl_double2 dPoint2 = { point2->x, point2->y };
#endif

  //Needs testing...
  int csResult = doCohenSutherlandTest(*point1, *point2, *minimum, *maximum);
  if (csResult != 0) return (csResult == 1) ? true : false;
  if (true) {
#if  DIM == 3
    cl_double3 normals[] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0,-1.0, };
#else 
    cl_double2 normals[] = { 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, };
#endif 
    cl_double2 temp = {0.0,0.0};
    double numerator, denominator;
    double width = maximum->x - minimum->x;

    //Else for each plane, check to see where the line crosses
    for (int i = 0; i < 4; ++i) {
      sub_v2v2(&temp, dMinimum, dPoint1);
      dot_v2v2(&numerator, temp, normals[i]);
      sub_v2v2(&temp, dPoint2, dPoint1);
      dot_v2v2(&denominator, temp, normals[i]);
      point_on_vn(&newPoint, dPoint1, temp, numerator / denominator);
      if (v2_on_aaedge(newPoint, dMinimum, normals[i], width)) return true;

      sub_v2v2(&temp, dMaximum, dPoint1);
      dot_v2v2(&numerator, temp, normals[i]);
      sub_v2v2(&temp, dPoint2, dPoint1);
      dot_v2v2(&denominator, temp, normals[i]);
      point_on_vn(&newPoint, dPoint1, temp, numerator / denominator);
      if (v2_on_aaedge(newPoint, dMinimum, normals[i], width)) return true;

    }
  }
  return false;
}