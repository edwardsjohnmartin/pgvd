__kernel void PointsToMortonKernel(
  __global BigUnsigned *inputBuffer,
  __global intn *points,
  const unsigned int size,
  const unsigned int bits
  ) 
{
 const size_t gid = get_global_id(0);
 const size_t lid = get_local_id(0);
 BigUnsigned tempBU;
 intn tempPoint = points[gid];

 if (gid < size) {
   xyz2z(&tempBU, tempPoint, bits);
 } else {
   initBlkBU(&tempBU, 0);
 }
 inputBuffer[gid] = tempBU;
 
}

__kernel void BitPredicateKernel( 
  __global BigUnsigned *inputBuffer, 
  __global Index *predicateBuffer, 
  Index index, 
  unsigned char comparedWith)
{
  BitPredicate(inputBuffer, predicateBuffer, index, comparedWith, get_global_id(0));
}

__kernel void UniquePredicateKernel(
 __global BigUnsigned *inputBuffer,
  __global Index *predicateBuffer)
{
  UniquePredicate(inputBuffer, predicateBuffer, get_global_id(0));
}

__kernel void StreamScanKernel( 
  __global Index* buffer, 
  __global Index* result, 
  __global volatile int* I, 
  __local Index* localBuffer, 
  __local Index* scratch)
{
  const size_t gid = get_global_id(0);
  const size_t lid = get_local_id(0);
  const size_t wid = get_group_id(0);
  const size_t ls = get_local_size(0);
  int sum = 0;  
  StreamScan_Init(buffer, localBuffer, scratch, gid, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = ls / 2; offset > 0; offset >>= 1) {
    AddAll(scratch, lid, offset);
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  //ADJACENT SYNCRONIZATION
  if (lid == 0 && gid != 0) {
  while (I[wid - 1] == -1);
  I[wid] = I[wid - 1] + scratch[0];
  }
  if (gid == 0) I[0] = scratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  result[gid] = scratch[0];

  scratch[lid] = localBuffer[lid];
  for (unsigned int i = 1; i < ls; i <<= 1) {
    HillesSteelScan(localBuffer, scratch, lid, i);
    __local Index *tmp = scratch;
    scratch = localBuffer;
    localBuffer = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  sum = localBuffer[lid];

  if (wid != 0) sum += I[wid - 1];
  result[gid] = sum; 
}

//Double Compaction
__kernel void BUCompactKernel( 
  __global BigUnsigned *inputBuffer, 
  __global BigUnsigned *resultBuffer, 
  __global Index *lPredicateBuffer, 
  __global Index *leftBuffer, 
  Index size)
{
  BUCompact(inputBuffer, resultBuffer, lPredicateBuffer, leftBuffer, size, get_global_id(0));
}

//Single Compaction
__kernel void BUSingleCompactKernel(
  __global BigUnsigned *inputBuffer,
  __global BigUnsigned *resultBuffer,
  __global Index *predicateBuffer,
  __global Index *addressBuffer)
{
  BUSingleCompact(inputBuffer, resultBuffer, predicateBuffer, addressBuffer, get_global_id(0));
}


//Binary Radix Tree Builder
__kernel void BuildBinaryRadixTreeKernel(
__global BrtNode *I,
__global BrtNode* L,
__global BigUnsigned* mpoints,
int mbits,
int size
) 
{
  BuildBinaryRadixTree(I, L, mpoints, mbits, size, get_global_id(0));
}


__kernel void ComputeLocalSplitsKernel(
  __global unsigned int* local_splits, 
  __global BrtNode* I,
  const int size
)
{
  const size_t gid = get_global_id(0);
  if (gid <size-1)
    local_splits[gid] = 0;
  if (gid == 0 && size > 0) 
    local_splits[0] = 1 + I[0].lcp_length / DIM;
  if ( gid < size - 1)
    ComputeLocalSplits(local_splits, I, gid);
}


//This is dumb and will be merged with the other kernel.
__kernel void BRT2OctreeKernel_init(
  __global BrtNode *I,
  __global OctNode *octree,
  __global unsigned int *local_splits,
  __global unsigned int *prefix_sums,
  const int n
) {
  const size_t gid = get_global_id(0);
  const int octree_size = prefix_sums[n-1];

  if (gid < octree_size)
    brt2octree_init( gid, octree);
}

__kernel void BRT2OctreeKernel(
  __global BrtNode *I,
  __global volatile OctNode *octree,
  __global unsigned int *local_splits,
  __global unsigned int *prefix_sums,
  const int n
) {
  const int gid = get_global_id(0);
  const int octree_size = prefix_sums[n-2];
  if (gid > 0 && gid < n-1){
    brt2octree( gid, I, octree, local_splits, prefix_sums, n, octree_size);
    octree[gid-1].pad1 = prefix_sums[gid-1];
  }  
}