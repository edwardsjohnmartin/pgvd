#pragma once
#include "KernelBox_.h"
#include "clfw.hpp"

namespace KernelBox {
  //
  //unsigned int GetSteamScanWorkGroupSize(unsigned int globalSize) {
  //  using namespace std;
  //  unsigned int localSize;
  //  clGetKernelWorkGroupInfo(Kernels["StreamScanKernel"], CLFW::Devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(unsigned int), &localSize, NULL);
  //  return min((int)(localSize), (int)globalSize);
  //}

  //cl_int initUniqueBuffers(int globalSize) {
  //  cl_int error = 0;
  //  if (!isBufferUsable(buffers.predicate, sizeof(Index)* (globalSize)))
  //    error |= createBuffer(buffers.predicate, sizeof(Index)*(globalSize));
  //  if (!isBufferUsable(buffers.address, sizeof(Index)* (globalSize)))
  //    error |= createBuffer(buffers.address, sizeof(Index)*(globalSize));
  //  unsigned int streamScanLocalSize = GetSteamScanWorkGroupSize(globalSize);
  //  if (!isBufferUsable(buffers.intermediate, sizeof(cl_int)*(globalSize / streamScanLocalSize)))
  //    error |= createBuffer(buffers.intermediate, sizeof(cl_int)*(globalSize / streamScanLocalSize));
  //  if (!isBufferUsable(buffers.bigUnsignedResult, sizeof(BigUnsigned)* (globalSize)))
  //    error |= createBuffer(buffers.bigUnsignedResult, sizeof(Index)*(globalSize));
  //  return error;
  //}

  //cl_int initBinaryRadixTreeBuffers(int globalSize) {
  //  cl_int error = 0;
  //  if (!isBufferUsable(buffers.internalNodes, sizeof(BrtNode)* (globalSize)))
  //    error |= createBuffer(buffers.internalNodes, sizeof(BrtNode)* (globalSize));
  //  if (!isBufferUsable(buffers.leafNodes, sizeof(BrtNode)* (globalSize)))
  //    error |= createBuffer(buffers.leafNodes, sizeof(BrtNode)* (globalSize));
  //  return error;
  //}

  //cl_int initBinaryRadixToOctreeBuffers(int globalSize) {
  //  cl_int error = 0;
  //  if (!isBufferUsable(buffers.localSplits, sizeof(unsigned int)* (globalSize)))
  //    error |= createBuffer(buffers.localSplits, sizeof(unsigned int)* (globalSize));
  //  if (!isBufferUsable(buffers.localSplitsCopy, sizeof(unsigned int)* (globalSize))) {
  //    error |= createBuffer(buffers.localSplitsCopy, sizeof(unsigned int)* (globalSize));
  //    unsigned int zero = 0;
  //    error |= clEnqueueFillBuffer(CLFW::Queues[0], buffers.localSplitsCopy->getBuffer(), 
  //      &zero, sizeof(unsigned int), 0, sizeof(unsigned int)* (globalSize), 0, NULL, NULL);
  //  }
  //  if (!isBufferUsable(buffers.scannedSplits, sizeof(unsigned int)* (globalSize)))
  //    error |= createBuffer(buffers.scannedSplits, sizeof(unsigned int)* (globalSize));
  //  return error;
  //}
}