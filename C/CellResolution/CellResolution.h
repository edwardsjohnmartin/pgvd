#pragma once
#ifndef __OPENCL_VERSION__
	#include "../Vector/vec_n.h"
#else
	#include "./OpenCL/C/Vector/vec_n.h"
#endif

/* Ambiguous cells code */
unsigned char computeOutCode(floatn point, floatn min, floatn max);
void sub_v2v2(cl_double2 *a, cl_double2 b, cl_double2 c);
void dot_v2v2(double *dot, cl_double2 a, cl_double2 b);
void point_on_vn(cl_double2 *result, cl_double2 point, cl_double2 ray, double t);
bool v3_on_aasquare(cl_float3 point, cl_float3 min, cl_float3 normal, float width);
bool v2_on_aaedge(cl_double2 point, cl_double2 min, cl_double2 normal, float width);
bool doLineBoxTest(floatn point1, floatn point2, floatn minimum, floatn maximum);
