__kernel
void reduce(__global unsigned int* buffer,
                __local unsigned int* scratch,
                __const int length,
                __global unsigned int* result) 
{
    //InitReduce
    int global_index = get_global_id(0);
    unsigned int accumulator = 0;

    // Loop sequentially over chunks of input vector
    // improves Big O by Brent's Theorem.
    while (global_index < length) {
        accumulator += buffer[global_index];
        global_index += get_global_size(0);
    }

    // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
        if (local_index < offset)
            scratch[local_index] = scratch[local_index] + scratch[local_index + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}