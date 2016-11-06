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

__kernel
void CheckOrder(
            __global BigUnsigned *zpoints,
            __local unsigned int* scratch,
            __const int numZPoints,
            __const int nextPowerOfTwo,
            __global unsigned int* result) 
{
    //Init
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);
    unsigned int accumulator = 0;

    // Loop sequentially over chunks of the BU vector, comparing along the way
    while(global_index < nextPowerOfTwo) {
        if(global_index < numZPoints-1)
        {
            BigUnsigned mine = zpoints[global_index];
            BigUnsigned other = zpoints[global_index + 1];
            accumulator += max(compareBU(&mine, &other), 0);
        }
        global_index += get_global_size(0);
        //Enforce coherency
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Perform parallel reduction
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

__kernel void GetTwoBitMaskKernel(
__global BigUnsigned *inputBuffer,
__global unsigned *masks,
__local BigUnsigned *localBUBuffer,
__local unsigned *localBoolBuffer,
unsigned int index,
unsigned char compared
)
{
  const size_t gid = get_global_id(0);
  const size_t lid = get_local_id(0);

  localBUBuffer[lid] = inputBuffer[gid];
  GetTwoBitMask(localBUBuffer, localBoolBuffer, index, compared, lid);

  masks[gid*4] = localBoolBuffer[lid*4];
  masks[gid*4 + 1] = localBoolBuffer[lid*4 + 1];
  masks[gid*4 + 2] = localBoolBuffer[lid*4 + 2];
  masks[gid*4 + 3] = localBoolBuffer[lid*4 + 3];
}