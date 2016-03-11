#include "CLWrapper.h"

//--PUBLIC--//
inline std::string buToString(BigUnsigned bu) {
	std::string representation = "";
	if (bu.len == 0)
	{
		representation += "[0]";
	}
	else {
		for (int i = bu.len; i > 0; --i) {
			representation += "[" + std::to_string(bu.blk[i - 1]) + "]";
		}
	}

	return representation;
}
void CLWrapper::RadixSort(void *input, const Index numBits, int _globalSize, int _localSize) {
  globalSize = _globalSize;
  localSize = _localSize;
  buffers.resize(8);
  //Input: 0. LPBuffer: 1. RPBuffer: 2. LABuffer: 3. RABuffer: 4. Result: 5.
  if (sizeof(BigUnsigned) > 0 && !(sizeof(BigUnsigned)& (sizeof(BigUnsigned)-1))) {
    initRadixSortBuffers(input);
    envokeRadixSortRoutine(numBits);
    checkError(clEnqueueReadBuffer(queue, buffers[0], CL_TRUE, 0, globalSize * sizeof (BigUnsigned), input, 0, nullptr, nullptr));

	for (int i = 0; i < buffers.size(); i++) {
		checkError(clReleaseMemObject(buffers[i]));
	}

	buffers.resize(0);
  }
  else {
    std::cout << "CLWrapper: Sorry, but BigUnsigned is " << std::to_string(sizeof(BigUnsigned)) <<" bytes, which isn't a power of two and cannot be sorted in parallel." << std::endl;
  }
};

//--PRIVATE--//
inline void CLWrapper::initRadixSortBuffers(void* input){
  //Input: 0. LPBuffer: 1. RPBuffer: 2. LABuffer: 3. RABuffer: 4. Result: 5. Intermediate. 6. ICopy. 7
  cl_int error = CL_SUCCESS;
  //std::vector<Index> zeroVector(globalSize);
  //std::vector<BigUnsigned> result(globalSize);

  buffers[0]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (BigUnsigned)* (globalSize), NULL, &error));
  error = clEnqueueWriteBuffer(queue, buffers[0], 1, 0, sizeof(BigUnsigned)* (globalSize), input, 0, nullptr, nullptr);

  buffers[1]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), NULL, &error));
  buffers[2]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), NULL, &error));
  buffers[3]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), NULL, &error));
  buffers[4]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), NULL, &error));
  buffers[5]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (BigUnsigned)* (globalSize), NULL, &error));

  //zeroVector.resize(globalSize / localSize);
  buffers[6] = (clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*(globalSize / localSize), NULL, &error));
  buffers[7]=(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*(globalSize / localSize), NULL, &error));

  if (error != CL_SUCCESS) {
	  std::cerr << "CLWrapper: Initializing radix sort buffers - OpenCL call failed with error " << error << std::endl;
	  std::getchar();
	  std::exit;
  }

  std::vector<cl_int> iValues(globalSize / localSize, -1);
  error = clEnqueueWriteBuffer(queue, buffers[6], 1, 0, sizeof(cl_int)*(globalSize / localSize), iValues.data(), 0, nullptr, nullptr);

  if (error != CL_SUCCESS) {
    std::cerr << "CLWrapper: Writing input to radix sort buffers - OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
}
#include <chrono>
void CLWrapper::envokeRadixSortRoutine(const Index numBits){
  //auto begin = std::chrono::high_resolution_clock::now();

  //Input: 0. LPBuffer: 1. RPBuffer: 2. LABuffer: 3. RABuffer: 4. Result: 5. Intermediate. 6. ICopy. 7
  //Using default workgroup size.
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  const size_t localWorkSize[] = { localSize, 0, 0 };

  //Creating the intermediate buffer for the streamscan.
  //cl_mem intermediateBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (cl_int)* (globalSize / localSize), nullptr, &error);
  cl_mem temp;
  std::vector<Index> predication(globalSize);
  //For each bit
  for (Index index = 0; index < numBits; index++) {
    //Predicate the 0's and 1's
    kernelBox->predicate(buffers[0], buffers[1], index, 1, globalSize, localSize);
    kernelBox->predicate(buffers[0], buffers[2], index, 0, globalSize, localSize);

    //Scan the predication buffers.
    kernelBox->streamScan(buffers[1], buffers[6], buffers[7], buffers[3], globalSize, localSize);
    kernelBox->streamScan(buffers[2], buffers[6], buffers[7], buffers[4], globalSize, localSize);
    
	//Compacting
    kernelBox->doubleCompact(buffers[0], buffers[5], buffers[1], buffers[3], buffers[4], globalSize, localSize);

    //Swap result with input.
	temp = buffers[0];
    buffers[0] = buffers[5];
    buffers[5] = temp;
  }
  //checkError(clReleaseMemObject(intermediateBuffer));
  //auto end = std::chrono::high_resolution_clock::now();
  //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
  //getchar();
  //std::cout << std::endl;
}


