#ifndef __CONFLICT_H__
#define __CONFLICT_H__

#ifdef __OPENCL_VERSION__
	#include "./SharedSources/Vector/vec.h"
	#include "./SharedSources/OctreeDefinitions/defs.h"
#elif defined __cplusplus
	#include "../Vector/vec.h"
	#include "../OctreeDefinitions/defs.h"

//I think defs.h might handle __local and __global...
	#define __local
	#define __global
#endif


//------------------------------------------------------------
// Usage:
//
// const int n = sample_conflict_count(
//     q0, q1, r0, r1, origin, width);
// floatn* samples = new floatn[n];
// floatn_array sample_array = make_floatn_array(samples);
// sample_conflict(
//     q0, q1, r0, r1, origin, width,
//     &sample_array, NULL, NULL);
//
// q0 and q1 are the endpoints of one line segment; r0 and r1
// are the endpoints of the other line segment. origin is the
// coordinate of the lower-left corner of the octree cell,
// and width is the width of the octree cell.
//
// After the above code has executed the sample points are
// in the samples variable.
//------------------------------------------------------------

// This struct does NOT have dynamic allocation or even bounds checking.
// Rather, it is a convenience for easily adding points to a pre-allocated
// array.
typedef struct floatn_array {
  floatn* array;
  int i;
} floatn_array;
inline floatn_array make_floatn_array(floatn* array) {
  floatn_array fa = { array, 0 };
  return fa;
}

int sample_conflict_count(
    const intn q0, const intn q1, const intn r0, const intn r1,
    const intn origin, const int width);

void sample_conflict(
    const intn q0, const intn q1, const intn r0, const intn r1,
    const intn origin, const int width, floatn_array* samples);

#endif
#ifndef __OPENCL_VERSION__
#undef __local
#undef __global
#endif
