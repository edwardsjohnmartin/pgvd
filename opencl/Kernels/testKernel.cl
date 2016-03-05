kernel void TestKernel(
	__global testStruct *inputBuffer,
	__global testStruct *resultBuffer,
	Index index
) {
	int gid = get_global_id(0);
	inputBuffer[gid].x = getOne();
	resultBuffer[gid] = inputBuffer[gid];
}