#include "CLWrapper.h"

//--PUBLIC--//
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


void* CLWrapper::getSharedMemoryPointer(size_t size, int readOrWriteFlag) {
	if (readOrWriteFlag == CL_MAP_WRITE) {
		sharedBuffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &error));
		if (error != CL_SUCCESS) {
			std::cerr << "CLWrapper: Creating shared memory - OpenCL call failed with error " << error << std::endl;
			std::getchar();
			std::exit;
		}
	} 
	void* pointer = clEnqueueMapBuffer(queue, sharedBuffers[sharedBuffers.size() - 1], CL_TRUE, readOrWriteFlag, 0, size, 0, NULL, NULL, &error);
	if (readOrWriteFlag == CL_MAP_WRITE) {
		sharedPointers.push_back(pointer);
	}
	return pointer;
}
void CLWrapper::unmapSharedMemory() {
	for (int i = 0; i < sharedBuffers.size(); i++) {
		checkError(clReleaseMemObject(sharedBuffers[i]));
	}
	sharedBuffers.resize(0);
	sharedPointers.resize(0);
}
void CLWrapper::RadixSort(int sharedMemoryIndex, const Index numBits, int _globalSize, int _localSize) {
  globalSize = _globalSize;
  localSize = _localSize;
  gpuBuffers.resize(6);

  //LPBuffer: 0. RPBuffer: 1. LABuffer: 2. RABuffer: 3. Result: 4. Intermediate: 5.
  if (sizeof(BigUnsigned) > 0 && !(sizeof(BigUnsigned)& (sizeof(BigUnsigned)-1))) {
    initRadixSortBuffers(sharedMemoryIndex);
    envokeRadixSortRoutine(numBits);
	  for (int i = 0; i < gpuBuffers.size(); i++) {
		  checkError(clReleaseMemObject(gpuBuffers[i]));
	  }
	  gpuBuffers.resize(0);
  }
  else 
    std::cout << "CLWrapper: Sorry, but BigUnsigned is " << std::to_string(sizeof(BigUnsigned)) 
      <<" bytes, which isn't a power of two and cannot be sorted in parallel." << std::endl;
};

int CLWrapper::UniqueSorted(int sharedMemoryIndex, int _globalSize, int _localSize) {
  globalSize = _globalSize;
  localSize = _localSize;
  gpuBuffers.resize(4);

  initUniqueBuffers(sharedMemoryIndex);
  kernelBox->uniquePredicate(sharedBuffers[sharedMemoryIndex], gpuBuffers[0], globalSize, localSize);
  kernelBox->streamScan(gpuBuffers[0], gpuBuffers[2], gpuBuffers[1], globalSize, localSize);
  
  kernelBox->singleCompact(sharedBuffers[sharedMemoryIndex], gpuBuffers[3], gpuBuffers[0], gpuBuffers[1], globalSize, localSize);
  

  cl_mem temp = sharedBuffers[sharedMemoryIndex];
  sharedBuffers[sharedMemoryIndex] = gpuBuffers[3];
  gpuBuffers[3] = temp;
  
  std::vector<Index> newSize(1);
  clEnqueueReadBuffer(queue, gpuBuffers[1], CL_TRUE, (sizeof(Index)*globalSize-2), sizeof(Index), newSize.data(), 0, NULL,NULL);

  for (int i = 0; i < gpuBuffers.size(); i++) {
    clReleaseMemObject(gpuBuffers[i]);
  }
 // std::cout << newSize[0] << std::endl;
  return newSize[0];
}

//--PRIVATE--//
inline void CLWrapper::initRadixSortBuffers(int sharedMemoryIndex){
  //LPBuffer: 0. RPBuffer: 1. LABuffer: 2. RABuffer: 3. Result: 4. Intermediate: 5.
  clEnqueueUnmapMemObject(queue, sharedBuffers[sharedMemoryIndex], sharedPointers[sharedMemoryIndex], 0, NULL, NULL);
  gpuBuffers[0]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), NULL, NULL));
  gpuBuffers[1]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), NULL, NULL));
  gpuBuffers[2]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), NULL, NULL));
  gpuBuffers[3]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), NULL, NULL));
  gpuBuffers[4]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (BigUnsigned)* (globalSize), NULL, NULL));
  gpuBuffers[5] = (clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*(globalSize / localSize), NULL, NULL));
}
void CLWrapper::envokeRadixSortRoutine(const Index numBits){
  //Input: 0. LPBuffer: 1. RPBuffer: 2. LABuffer: 3. RABuffer: 4. Result: 5.
  //Using default workgroup size.
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  const size_t localWorkSize[] = { localSize, 0, 0 };

  cl_mem temp;
  //For each bit
  for (Index index = 0; index < numBits; index++) {
    //Predicate the 0's and 1's
    kernelBox->bitPredicate(sharedBuffers[0], gpuBuffers[0], index, 1, globalSize, localSize);
    kernelBox->bitPredicate(sharedBuffers[0], gpuBuffers[1], index, 0, globalSize, localSize);

    //Scan the predication buffers.
    kernelBox->streamScan(gpuBuffers[0], gpuBuffers[5], gpuBuffers[2], globalSize, localSize);
    kernelBox->streamScan(gpuBuffers[1], gpuBuffers[5], gpuBuffers[3], globalSize, localSize);
    
	  //Compacting
    kernelBox->doubleCompact(sharedBuffers[0], gpuBuffers[4], gpuBuffers[0], gpuBuffers[2], gpuBuffers[3], globalSize, localSize);

    //Swap result with input.
	  temp = sharedBuffers[0];
	  sharedBuffers[0] = gpuBuffers[4];
	  gpuBuffers[4] = temp;
  }
}

inline void CLWrapper::initUniqueBuffers(int sharedMemoryIndex) {
  // PBuffer: 0. ABuffer: 1. Intermediate: 2 Result: 3
  //clEnqueueUnmapMemObject(queue, sharedBuffers[sharedMemoryIndex], sharedPointers[sharedMemoryIndex], 0, NULL, NULL);
  gpuBuffers[0] = (clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Index)* (globalSize), NULL, NULL));
  gpuBuffers[1] = (clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Index)* (globalSize), NULL, NULL));
  gpuBuffers[2] = (clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Index)* (globalSize), NULL, NULL));
  gpuBuffers[3] = (clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(BigUnsigned)* (globalSize), NULL, NULL));

}

