#include "CLWrapper.h"
#include <math.h>

//--PUBLIC--//
void CLWrapper::RadixSort(const vector<intn>& points, const int bits, const Index mBits) {
  globalSize = max((int)pow(2, ceil(log(points.size()) / log(2))), 8); //buffers.bigUnsignedInput->getSize()/sizeof(BigUnsigned);
  localSize = min((int)globalSize, 256);//min((int)pow(2, floor(log(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)/log(2))), (int)globalSize);

  //LPBuffer: 0. RPBuffer: 1. LABuffer: 2. RABuffer: 3. Result: 4. Intermediate: 5.
  if (sizeof(BigUnsigned) > 0 && !(sizeof(BigUnsigned)& (sizeof(BigUnsigned)-1))) {
    initRadixSortBuffers();

    clEnqueueWriteBuffer(queue, buffers.points->getBuffer(), true, 0, points.size() * sizeof(intn), points.data(), 0, NULL, NULL);

    envokeRadixSortRoutine(points.size(), bits, mBits);
  }
  else 
    std::cout << "CLWrapper: Sorry, but BigUnsigned is " << std::to_string(sizeof(BigUnsigned)) 
      <<" bytes, which isn't a power of two and cannot be sorted in parallel." << std::endl;
};
size_t CLWrapper::UniqueSorted() {
  globalSize = buffers.bigUnsignedInput->getSize()/sizeof(BigUnsigned);
  
  initUniqueBuffers();
  kernelBox->uniquePredicate(buffers.bigUnsignedInput->getBuffer(), buffers.leftPredicate->getBuffer(), globalSize);
  kernelBox->streamScan(buffers.leftPredicate->getBuffer(), buffers.intermediate->getBuffer(), buffers.leftAddress->getBuffer(), globalSize);
  
  kernelBox->singleCompact(buffers.bigUnsignedInput->getBuffer(), buffers.bigUnsignedResult->getBuffer(), buffers.leftPredicate->getBuffer(), buffers.leftAddress->getBuffer(), globalSize);
  
  shared_ptr<Buffer> temp = buffers.bigUnsignedInput;
  buffers.bigUnsignedInput = buffers.bigUnsignedResult;
  buffers.bigUnsignedResult = temp;
  
  Index value;
  clEnqueueReadBuffer(queue, buffers.leftAddress->getBuffer(), CL_TRUE, (sizeof(Index)*globalSize-(sizeof(Index))), sizeof(Index), &value, 0, NULL,NULL);

 // std::cout << newSize[0] << std::endl;
  return value;
}
void CLWrapper::buildBrt(size_t n, int mbits) {
  globalSize = buffers.bigUnsignedInput->getSize() / sizeof(BigUnsigned);
  initBrtBuffers();
  kernelBox->buildBinaryRadixTree(buffers.internalNodes->getBuffer(), buffers.leafNodes->getBuffer(), buffers.bigUnsignedInput->getBuffer(), mbits, n, globalSize);
	// Note that it loops only n-1 times.
	//for (int i = 0; i < n - 1; ++i) {
//		build_brt_kernel(i, I, L, mpoints, n, resln);
//	}
}
vector<OctNode> CLWrapper::BRT2Octree(size_t n) {
  vector<unsigned int> local_splits_vec(n - 1, 0); // be sure to initialize to zero
  vector<unsigned int> prefix_sums_vec(n);
  int nextPowerOfTwo = max((int)pow(2, ceil(log(n) / log(2))), 8);
  initBRT2OctreeBuffers(nextPowerOfTwo);

  //compute local splits
  kernelBox->computeLocalSplits(buffers.localSplits->getBuffer(), buffers.internalNodes->getBuffer(), n, nextPowerOfTwo);

  //scan the splits
  kernelBox->streamScan(buffers.localSplits->getBuffer(), buffers.intermediate->getBuffer(), buffers.scannedSplits->getBuffer(), nextPowerOfTwo);
  
  //Read in the required octree size
  unsigned int octree_size; //= prefix_sums[n - 1];
  vector<unsigned int> sums(n,0);
  clEnqueueReadBuffer(queue, buffers.scannedSplits->getBuffer(), CL_TRUE, sizeof(unsigned int)*(n-2), sizeof(unsigned int), &octree_size, 0, NULL, NULL);
  
  //octree_size = sums[n - 1];
  //Make it a power of two.
  const int nextOctreeSizePowerOfTwo = max((int)pow(2, ceil(log(octree_size) / log(2))), 8);
  //Create an octree buffer.
  if (!isBufferUsable(buffers.octree, sizeof(OctNode)* (nextOctreeSizePowerOfTwo))) {
    buffers.octree = createBuffer(sizeof(OctNode)* (nextOctreeSizePowerOfTwo));
  }
  //use the scanned splits & brt to create octree.
  kernelBox->brt2Octree_init(buffers.internalNodes->getBuffer(), buffers.octree->getBuffer(), buffers.localSplits->getBuffer(), buffers.scannedSplits->getBuffer(), n, octree_size);
  kernelBox->brt2Octree(buffers.internalNodes->getBuffer(), buffers.octree->getBuffer(), buffers.localSplits->getBuffer(), buffers.scannedSplits->getBuffer(), n, octree_size);
  
  vector<OctNode> octree_vec(octree_size);

  clEnqueueReadBuffer(queue, buffers.octree->getBuffer(), true, 0, sizeof(OctNode)*octree_size, octree_vec.data(), 0, NULL, NULL);
  return octree_vec;
}

/*void CLWrapper::BuildBRT(int mpointsSMI, std::vector<BrtNode>& intermediates, std::vector<BrtNode>& leaves) {
  //Sort the mpoints
  //Unique the mpoints
  //

}
*/
//--PRIVATE--//
inline void CLWrapper::initRadixSortBuffers(){
  if (!isBufferUsable(buffers.points, sizeof(intn)* (globalSize)))
    buffers.points = createBuffer(sizeof(intn)* (globalSize));
  if (!isBufferUsable(buffers.bigUnsignedInput, sizeof(BigUnsigned)* (globalSize)))
    buffers.bigUnsignedInput = createBuffer(sizeof(BigUnsigned)* (globalSize));
  if (!isBufferUsable(buffers.leftPredicate, sizeof(Index)* (globalSize)))
    buffers.leftPredicate = createBuffer(sizeof (Index)* (globalSize));
  if (!isBufferUsable(buffers.rightPredicate, sizeof(Index)* (globalSize)))
    buffers.rightPredicate = createBuffer(sizeof(Index)*(globalSize));
  if (!isBufferUsable(buffers.leftAddress, sizeof(Index)* (globalSize)))
    buffers.leftAddress = createBuffer(sizeof(Index)*(globalSize));
  if (!isBufferUsable(buffers.rightAddress, sizeof(Index)* (globalSize)))
    buffers.rightAddress = createBuffer(sizeof(Index)*(globalSize));
  if (!isBufferUsable(buffers.bigUnsignedResult, sizeof(BigUnsigned)* (globalSize)))
     buffers.bigUnsignedResult = createBuffer(sizeof(BigUnsigned)*globalSize);
  //get local size from kernel
  size_t streamScanLocalSize = kernelBox->getSteamScanWorkGroupSize(globalSize);
  if (!isBufferUsable(buffers.intermediate, sizeof(cl_int)*(globalSize / streamScanLocalSize)))
    buffers.intermediate = createBuffer(sizeof(cl_int)*(globalSize / streamScanLocalSize));
}
void CLWrapper::envokeRadixSortRoutine(const int size, const int bits, const Index mbits){
  //Input: 0. LPBuffer: 1. RPBuffer: 2. LABuffer: 3. RABuffer: 4. Result: 5.
  //Using default workgroup size.
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  const size_t localWorkSize[] = { localSize, 0, 0 };
  
  //convert points to mpoints.
  kernelBox->pointsToMorton(buffers.bigUnsignedInput->getBuffer(), buffers.points->getBuffer(), size, bits, globalSize);

  shared_ptr<Buffer> temp;
  //For each bit
  for (Index index = 0; index < mbits; index++) {
    //Predicate the 0's and 1's
    cl_mem x=buffers.bigUnsignedInput->getBuffer();
    kernelBox->bitPredicate(buffers.bigUnsignedInput->getBuffer(), buffers.leftPredicate->getBuffer(), index, 1, globalSize);
    kernelBox->bitPredicate(buffers.bigUnsignedInput->getBuffer(), buffers.rightPredicate->getBuffer(), index, 0, globalSize);

    //Scan the predication buffers.
    kernelBox->streamScan(buffers.leftPredicate->getBuffer(), buffers.intermediate->getBuffer(), buffers.leftAddress->getBuffer(), globalSize);
    kernelBox->streamScan(buffers.rightPredicate->getBuffer(), buffers.intermediate->getBuffer(), buffers.rightAddress->getBuffer(), globalSize);
    
	  //Compacting
    kernelBox->doubleCompact(buffers.bigUnsignedInput->getBuffer(), buffers.bigUnsignedResult->getBuffer(), buffers.leftPredicate->getBuffer(), buffers.leftAddress->getBuffer(), buffers.rightAddress->getBuffer(), globalSize);

    //Swap result with input.
    temp = buffers.bigUnsignedInput;
    buffers.bigUnsignedInput = buffers.bigUnsignedResult;
    buffers.bigUnsignedResult = temp;
  }
}
inline void CLWrapper::initUniqueBuffers() {
  // PBuffer: 0. ABuffer: 1. Intermediate: 2 Result: 3
  //clEnqueueUnmapMemObject(queue, sharedBuffers[sharedMemoryIndex], sharedPointers[sharedMemoryIndex], 0, NULL, NULL);
  if (!isBufferUsable(buffers.leftPredicate, sizeof(Index)* (globalSize)))
    buffers.leftPredicate = createBuffer(sizeof(Index)*(globalSize));
  if (!isBufferUsable(buffers.leftAddress, sizeof(Index)* (globalSize)))
    buffers.leftAddress = createBuffer(sizeof(Index)*(globalSize));
  size_t streamScanLocalSize = kernelBox->getSteamScanWorkGroupSize(globalSize);
  if (!isBufferUsable(buffers.intermediate, sizeof(cl_int)*(globalSize / streamScanLocalSize)))
    buffers.intermediate = createBuffer(sizeof(cl_int)*(globalSize / streamScanLocalSize));
  if (!isBufferUsable(buffers.bigUnsignedResult, sizeof(BigUnsigned)* (globalSize)))
    buffers.bigUnsignedResult = createBuffer(sizeof(Index)*(globalSize));
}
inline void CLWrapper::initBrtBuffers() {
  if (!isBufferUsable(buffers.internalNodes, sizeof(BrtNode)* (globalSize)))
    buffers.internalNodes = createBuffer(sizeof(BrtNode)* (globalSize));
  if (!isBufferUsable(buffers.leafNodes, sizeof(BrtNode)* (globalSize)))
    buffers.leafNodes = createBuffer(sizeof(BrtNode)* (globalSize));
}
inline void CLWrapper::initBRT2OctreeBuffers(size_t n) {
  if (!isBufferUsable(buffers.localSplits, sizeof(unsigned int)* (n)))
    buffers.localSplits = createBuffer(sizeof(unsigned int)* (n));
  if (!isBufferUsable(buffers.scannedSplits, sizeof(unsigned int)* (n)))
    buffers.scannedSplits = createBuffer(sizeof(unsigned int)* (n));
}
