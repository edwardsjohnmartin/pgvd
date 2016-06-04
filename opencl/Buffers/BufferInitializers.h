#pragma once
#include "KernelBox_.h"
#include "clfw.hpp"

namespace KernelBox {
  
  size_t GetSteamScanWorkGroupSize(size_t globalSize) {
    using namespace std;
    size_t localSize;
    clGetKernelWorkGroupInfo(Kernels["StreamScanKernel"], CLFW::Devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &localSize, NULL);
    return min((int)(localSize), (int)globalSize);
  }

  cl_int initRadixSortBuffers(int globalSize) {
    cl_int error = 0;
    if (!isBufferUsable(buffers.predicate, sizeof(Index)* (globalSize)))
      error |= createBuffer(buffers.predicate, sizeof(Index)* (globalSize));
    if (!isBufferUsable(buffers.address, sizeof(Index)* (globalSize)))
      error |= createBuffer(buffers.address, sizeof(Index)*(globalSize));
    if (!isBufferUsable(buffers.bigUnsignedResult, sizeof(BigUnsigned)* (globalSize)))
      error |= createBuffer(buffers.bigUnsignedResult, sizeof(BigUnsigned)*globalSize);
    if (!isBufferUsable(buffers.bigUnsignedResultCopy, sizeof(BigUnsigned)* (globalSize))) {
      error |= createBuffer(buffers.bigUnsignedResultCopy, sizeof(BigUnsigned)*globalSize);
      BigUnsigned zero;
      initBlkBU(&zero, 0);
      error |= clEnqueueFillBuffer(CLFW::Queues[0], buffers.bigUnsignedResultCopy->getBuffer(), &zero, sizeof(BigUnsigned), 0, sizeof(BigUnsigned)* (globalSize), 0, NULL, NULL);
    }

    //get local size from kernel
    size_t streamScanLocalSize = GetSteamScanWorkGroupSize(globalSize);
    if (!isBufferUsable(buffers.intermediate, sizeof(cl_int)*(globalSize / streamScanLocalSize)))
      error |= createBuffer(buffers.intermediate, sizeof(cl_int)*(globalSize / streamScanLocalSize));
    if (!isBufferUsable(buffers.intermediateCopy, sizeof(cl_int)*(globalSize / streamScanLocalSize))) {
      error |= createBuffer(buffers.intermediateCopy, sizeof(cl_int)*(globalSize / streamScanLocalSize));
      Index negativeOne = -1;
      error |= clEnqueueFillBuffer(CLFW::Queues[0], buffers.intermediateCopy->getBuffer(), &negativeOne, sizeof(Index), 0, sizeof(Index) * globalSize / streamScanLocalSize, 0, NULL, NULL);
    }
    return error;
  }

}