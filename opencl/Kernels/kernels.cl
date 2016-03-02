//Radix Predication
__kernel void Predicate(
	__global BigUnsigned *inputBuffer,
	__global Index *predicateBuffer,
	Index index,
	unsigned char comparedWith)
{
	const size_t gid = get_global_id(0);
	BigUnsigned self = inputBuffer[gid];
  predicateBuffer[gid] = (getBUBit(&self, index) == comparedWith) ? 1:0;
}

//StreamScan
//https://www.youtube.com/watch?v=RdfmxfZBHpo
#define SWAP(a,b) {__local Index *tmp=a;a=b;b=tmp;}
__kernel
void StreamScan(
__global Index* buffer,
__global Index* result,
__global volatile int* I,
__local int* localBuffer,
__local int* scratch)
{
	//INITIALIZATION
  const size_t gid = get_global_id(0);
  const size_t lid = get_local_id(0);
  const size_t wid = get_group_id(0);
  const size_t ls = get_local_size(0);
  Index sum;
	localBuffer[lid] = scratch[lid] = buffer[gid];
  barrier(CLK_LOCAL_MEM_FENCE);

	//ADDITIVE REDUCTION
	for (int offset = ls / 2; offset > 0; offset >>= 1) {
    if (lid < offset){ scratch[lid] = scratch[lid + offset] + scratch[lid];}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//ADJACENT SYNCRONIZATION
  if (lid == 0 && gid != 0) {
    while (I[wid - 1] == -1);
    I[wid] = I[wid - 1] + scratch[0];
  }
  if (gid == 0) I[0] = scratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  //SCAN
  scratch[lid] = localBuffer[lid];
	for (uint i = 1; i < ls; i <<= 1) {
		if (lid >(i - 1))
			scratch[lid] = localBuffer[lid] + localBuffer[lid - i];
		else
			scratch[lid] = localBuffer[lid];
	  SWAP(scratch, localBuffer);
	  barrier(CLK_LOCAL_MEM_FENCE);
	}
  sum = localBuffer[lid];
  if (wid != 0) sum += I[wid - 1];
  result[gid] = sum;
}

//Double Compaction
__kernel void BUCompact(
	__global BigUnsigned *inputBuffer,
  __global BigUnsigned *resultBuffer,
	__global Index *lPredicateBuffer,
	__global Index *leftBuffer,
	__global Index *rightBuffer,
	Index size)
{
	const size_t gid = get_global_id(0);
	Index index;
	if (lPredicateBuffer[gid] == 1) index = leftBuffer[gid];
	else index = rightBuffer[gid] + leftBuffer[size - 1];
	BigUnsigned temp = inputBuffer[gid];
  resultBuffer[index - 1] = temp;
}
