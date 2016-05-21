#include "CLWrapper.h"

extern "C" {
#include "../../C/BuildOctree.h"
}
inline std::string buToString(BigUnsigned bu) {
  std::string representation = "";
  if (bu.len == 0)
  {
    if (bu.isNULL) {
      representation += "[NULL]";
    }
    else {
      representation += "[0]";
    }
  }
  else {
    for (int i = bu.len; i > 0; --i) {
      representation += "[" + std::to_string(bu.blk[i - 1]) + "]";
    }
  }

  return representation;
}

//--PUBLIC--//
void CLWrapper::RadixSort(const vector<intn>& points, const int bits, const Index mBits) {
  globalSize = max((int)pow(2, ceil(log(points.size()) / log(2))), 8); //buffers.bigUnsignedInput->getSize()/sizeof(BigUnsigned);
  localSize = min((int)globalSize, 256);//min((int)pow(2, floor(log(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)/log(2))), (int)globalSize);

  //LPBuffer: 0. RPBuffer: 1. LABuffer: 2. RABuffer: 3. Result: 4. Intermediate: 5.
  if (sizeof(BigUnsigned) > 0 && !(sizeof(BigUnsigned)& (sizeof(BigUnsigned)-1))) {
    initRadixSortBuffers();

    intn* gpuPoints = (intn*)buffers.points->map_buffer();
    memcpy(gpuPoints, points.data(), points.size() * sizeof(intn));
    buffers.points->unmap_buffer();
    
    envokeRadixSortRoutine(points.size(), bits, mBits);
  }
  else 
    std::cout << "CLWrapper: Sorry, but BigUnsigned is " << std::to_string(sizeof(BigUnsigned)) 
      <<" bytes, which isn't a power of two and cannot be sorted in parallel." << std::endl;
};
size_t CLWrapper::UniqueSorted() {
  globalSize = buffers.bigUnsignedInput->getSize()/sizeof(BigUnsigned);
  
  initUniqueBuffers();
  kernelBox->uniquePredicate(buffers.bigUnsignedInput->getBuffer(), buffers.predicate->getBuffer(), globalSize);
  kernelBox->streamScan(buffers.predicate->getBuffer(), buffers.intermediate->getBuffer(), buffers.intermediateCopy->getBuffer(), buffers.address->getBuffer(), globalSize);
  
  kernelBox->singleCompact(buffers.bigUnsignedInput->getBuffer(), buffers.bigUnsignedResult->getBuffer(), buffers.predicate->getBuffer(), buffers.address->getBuffer(), globalSize);
  
  shared_ptr<Buffer> temp = buffers.bigUnsignedInput;
  buffers.bigUnsignedInput = buffers.bigUnsignedResult;
  buffers.bigUnsignedResult = temp;
  
  Index value;
  clEnqueueReadBuffer(queue, buffers.address->getBuffer(), CL_TRUE, (sizeof(Index)*globalSize-(sizeof(Index))), sizeof(Index), &value, 0, NULL,NULL);

  return value;
}
void CLWrapper::buildBrt(size_t n, int mbits) {
  globalSize = buffers.bigUnsignedInput->getSize() / sizeof(BigUnsigned);
  initBrtBuffers();
  kernelBox->buildBinaryRadixTree(buffers.internalNodes->getBuffer(), buffers.leafNodes->getBuffer(), buffers.bigUnsignedInput->getBuffer(), mbits, n, globalSize);
}

void CLWrapper::BRT2Octree(size_t n, vector<OctNode> &octree_vec) {
  vector<unsigned int> local_splits_vec(n - 1, 0); // be sure to initialize to zero
  vector<unsigned int> prefix_sums_vec(n);
  int nextPowerOfTwo = max((int)pow(2, ceil(log(n) / log(2))), 8);
  initBRT2OctreeBuffers(nextPowerOfTwo);

  //compute local splits
  kernelBox->computeLocalSplits(buffers.localSplits->getBuffer(), buffers.internalNodes->getBuffer(), n, nextPowerOfTwo);
  
  //scan the splits
  kernelBox->streamScan(buffers.localSplits->getBuffer(), buffers.intermediate->getBuffer(), buffers.intermediateCopy->getBuffer(), buffers.scannedSplits->getBuffer(), nextPowerOfTwo);

  //Read in the required octree size
  int octree_size; //= prefix_sums[n - 1];
  clEnqueueReadBuffer(queue, buffers.scannedSplits->getBuffer(), CL_TRUE, sizeof(int)*(n-2), sizeof(int), &octree_size, 0, NULL, NULL);
  
  //Make it a power of two.
  const int nextOctreeSizePowerOfTwo = max((int)pow(2, ceil(log(octree_size) / log(2))), 8);
  
  //Create an octree buffer.
  if (!isBufferUsable(buffers.octree, sizeof(OctNode)* (nextOctreeSizePowerOfTwo))) {
    buffers.octree = createBuffer(sizeof(OctNode)* (nextOctreeSizePowerOfTwo));
  }
  //use the scanned splits & brt to create octree.
  kernelBox->brt2Octree_init(buffers.internalNodes->getBuffer(), buffers.octree->getBuffer(), buffers.localSplits->getBuffer(), buffers.scannedSplits->getBuffer(), n, nextOctreeSizePowerOfTwo);
  kernelBox->brt2Octree(buffers.internalNodes->getBuffer(), buffers.octree->getBuffer(), buffers.localSplits->getBuffer(), buffers.scannedSplits->getBuffer(), n, n);
  octree_vec.resize(octree_size);
  clEnqueueReadBuffer(queue, buffers.octree->getBuffer(), true, 0, sizeof(OctNode)*(octree_size), octree_vec.data(), 0, NULL, NULL);
}

//--PRIVATE--//
inline void CLWrapper::initRadixSortBuffers(){
  if (!isBufferUsable(buffers.points, sizeof(intn)* (globalSize)))
    buffers.points = createBuffer(sizeof(intn)* (globalSize));
  if (!isBufferUsable(buffers.bigUnsignedInput, sizeof(BigUnsigned)* (globalSize)))
    buffers.bigUnsignedInput = createBuffer(sizeof(BigUnsigned)* (globalSize));
  if (!isBufferUsable(buffers.predicate, sizeof(Index)* (globalSize)))
    buffers.predicate = createBuffer(sizeof (Index)* (globalSize));
  if (!isBufferUsable(buffers.address, sizeof(Index)* (globalSize)))
    buffers.address = createBuffer(sizeof(Index)*(globalSize));
  if (!isBufferUsable(buffers.bigUnsignedResult, sizeof(BigUnsigned)* (globalSize)))
     buffers.bigUnsignedResult = createBuffer(sizeof(BigUnsigned)*globalSize);
  if (!isBufferUsable(buffers.bigUnsignedResultCopy, sizeof(BigUnsigned)* (globalSize))) {
    buffers.bigUnsignedResultCopy = createBuffer(sizeof(BigUnsigned)*globalSize);
    BigUnsigned zero;
    initBlkBU(&zero, 0);
    clEnqueueFillBuffer(queue, buffers.bigUnsignedResultCopy->getBuffer(), &zero, sizeof(BigUnsigned), 0, sizeof(BigUnsigned)* (globalSize), 0, NULL, NULL);
  }

  //get local size from kernel
  size_t streamScanLocalSize = kernelBox->getSteamScanWorkGroupSize(globalSize);
  if (!isBufferUsable(buffers.intermediate, sizeof(cl_int)*(globalSize / streamScanLocalSize)))
    buffers.intermediate = createBuffer(sizeof(cl_int)*(globalSize / streamScanLocalSize));
  if (!isBufferUsable(buffers.intermediateCopy, sizeof(cl_int)*(globalSize / streamScanLocalSize))) {
    buffers.intermediateCopy = createBuffer(sizeof(cl_int)*(globalSize / streamScanLocalSize));
    Index negativeOne = -1;
    clEnqueueFillBuffer(queue, buffers.intermediateCopy->getBuffer(), &negativeOne, sizeof(Index), 0, sizeof(Index) * globalSize/streamScanLocalSize, 0, NULL, NULL);
  }

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
    kernelBox->bitPredicate(buffers.bigUnsignedInput->getBuffer(), buffers.predicate->getBuffer(), index, 1, globalSize);

    //Scan the predication buffers.
    kernelBox->streamScan(buffers.predicate->getBuffer(), buffers.intermediate->getBuffer(), buffers.intermediateCopy->getBuffer(), buffers.address->getBuffer(), globalSize);

	  //Compacting
    kernelBox->doubleCompact(buffers.bigUnsignedInput->getBuffer(), buffers.bigUnsignedResult->getBuffer(), buffers.bigUnsignedResultCopy->getBuffer(), buffers.predicate->getBuffer(), buffers.address->getBuffer(), globalSize);

    //Swap result with input.
    temp = buffers.bigUnsignedInput;
    buffers.bigUnsignedInput = buffers.bigUnsignedResult;
    buffers.bigUnsignedResult = temp;
  }
}
inline void CLWrapper::initUniqueBuffers() {
  // PBuffer: 0. ABuffer: 1. Intermediate: 2 Result: 3
  if (!isBufferUsable(buffers.predicate, sizeof(Index)* (globalSize)))
    buffers.predicate = createBuffer(sizeof(Index)*(globalSize));
  if (!isBufferUsable(buffers.address, sizeof(Index)* (globalSize)))
    buffers.address = createBuffer(sizeof(Index)*(globalSize));
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
