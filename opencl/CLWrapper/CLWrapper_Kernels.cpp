#include "CLWrapper.h"

//--PUBLIC--//
void CLWrapper::RadixSort(std::vector<BigUnsigned> &input, const Index numBits) {
  //Input: 0. LPBuffer: 1. RPBuffer: 2. LABuffer: 3. RABuffer: 4. Result: 5.
  if (sizeof(BigUnsigned) > 0 && !(sizeof(BigUnsigned)& (sizeof(BigUnsigned)-1))) {
    initRadixSortBuffers(input);
    envokeRadixSortRoutine(numBits);
    checkError(clEnqueueReadBuffer(queue, buffers[0], CL_TRUE, 0, globalSize * sizeof (BigUnsigned), input.data(), 0, nullptr, nullptr));
  }
  else {
    std::cout << "CLWrapper: Sorry, but BigUnsigned is " << std::to_string(sizeof(BigUnsigned)) <<" bytes, which isn't a power of two and cannot be sorted in parallel." << std::endl;
  }
};

//--PRIVATE--//
void CLWrapper::initRadixSortBuffers(std::vector<BigUnsigned> &input){
  //Input: 0. LPBuffer: 1. RPBuffer: 2. LABuffer: 3. RABuffer: 4. Result: 5. Intermediate. 6. ICopy. 7
  cl_int error = CL_SUCCESS;
  std::vector<Index> zeroVector(globalSize);
  std::vector<BigUnsigned> result(globalSize);

  buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof (BigUnsigned)* (globalSize), input.data(), &error));
  buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), zeroVector.data(), &error));
  buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), zeroVector.data(), &error));
  buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), zeroVector.data(), &error));
  buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (Index)* (globalSize), zeroVector.data(), &error));
  buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof (BigUnsigned)* (globalSize), result.data(), &error));

  zeroVector.resize(globalSize / localSize);
  buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*(globalSize / localSize), zeroVector.data(), &error));
  buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*(globalSize / localSize), zeroVector.data(), &error));

  std::vector<cl_int> iValues(globalSize / localSize, -1);
  clEnqueueWriteBuffer(queue, buffers[6], 1, 0, sizeof(cl_int)*(globalSize / localSize), iValues.data(), 0, nullptr, nullptr);

  if (error != CL_SUCCESS) {
    std::cerr << "CLWrapper: Initializing radix sort buffers - OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
}
#include <chrono>
void CLWrapper::envokeRadixSortRoutine(const Index numBits){
  auto begin = std::chrono::high_resolution_clock::now();

  //Input: 0. LPBuffer: 1. RPBuffer: 2. LABuffer: 3. RABuffer: 4. Result: 5. Intermediate. 6. ICopy. 7
  //Using default workgroup size.
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  const size_t localWorkSize[] = { localSize, 0, 0 };

  //Creating the intermediate buffer for the streamscan.
  cl_mem intermediateBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof (cl_int)* (globalSize / localSize), nullptr, &error);
  
  //For each bit
  for (Index index = 0; index < numBits; index++) {
    //Predicate the 0's and 1's
    kernelBox->predicate(buffers[0], buffers[1], index, 0, globalSize, localSize);
    kernelBox->predicate(buffers[0], buffers[2], index, 1, globalSize, localSize);

    //Scan the predication buffers.
    kernelBox->streamScan(buffers[1], buffers[6], buffers[7], buffers[3], globalSize, localSize);
    kernelBox->streamScan(buffers[2], buffers[6], buffers[7], buffers[4], globalSize, localSize);
    
    //Compacting
    kernelBox->doubleCompact(buffers[0], buffers[5], buffers[1], buffers[3], buffers[4], globalSize, localSize);

    //Swap result with input.
    cl_mem temp = buffers[0];
    buffers[0] = buffers[5];
    buffers[5] = temp;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
  //getchar();

}


