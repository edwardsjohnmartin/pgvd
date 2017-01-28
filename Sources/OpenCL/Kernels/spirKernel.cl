__kernel void test
(
__global int* input,
__global int* output
)
{
	output[get_global_id(0)] = input[get_global_id(0)] * 5;
} 