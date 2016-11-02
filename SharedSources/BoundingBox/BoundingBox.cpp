
#ifdef __OPENCL_VERSION__ 
#include "./SharedSources/BoundingBox/BoundingBox.h"
#else
#include "./BoundingBox.h"
#endif

void do_stuff(const floatn* v) {
	printf("%d\n", v->x);
}