#pragma once
using namespace std;
#include "../Kernels/KernelBox.h"
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include <CL/cl.hpp>
#endif // __APPLE__

#include <vector>
#include <iostream>
#include <memory>
#include <algorithm>
#include "Buffers/Buffers.h"
#include "Buffers/Buffer.h"
#include "C/BrtNode.h"
#include "OctNode.h"
#include "opencl/vec.h"
#include "timer.h"

/*
  CLWrapper.
    -Handles all OpenCL Initialization
    -Provides parallel computation wrappers

  Nate B/V - bitinat2@isu.edu
  2/13/2016
*/
class CLWrapper
{
  private:
	  //Variables
	  cl_int error;
	  cl_uint platformIdCount;
	  cl_uint deviceIdCount;
	  cl_context context;
	  cl_command_queue queue;
	  vector<cl_platform_id> platformIds;
	  vector<cl_device_id> deviceIds;
	  KernelBox* kernelBox;
    Timer timer;

	  //Initializers
	  void initPlatformIds();
	  void initDevices();
	  void initContext();
	  void initCommandQueue();
	  void initKernelBox();

	  //Private helper methods
	  string getPlatformName(cl_platform_id id);
	  string getDeviceName(cl_device_id id);
	  void initRadixSortBuffers();
	  void envokeRadixSortRoutine(const int size, const Index mbits);
	  void initUniqueBuffers();
	  void InitBinaryRadixTreeBuffers();
    void InitBinaryRadixToOctreeBuffers(size_t n);
	  void checkError(cl_int error);

  public:
    Buffers buffers;
	  bool verbose = true;
	  size_t globalSize;
	  size_t localSize;

	  CLWrapper(size_t globalSize, size_t localSize);
	  ~CLWrapper();
    //Public helper methods
    shared_ptr<Buffer> createBuffer(size_t size);
    bool isBufferUsable(shared_ptr<Buffer> buffer, size_t expectedSizeInBytes);
	  
    //Kernel Wrappers
    void UploadPoints(const vector<intn>& points);
    void DownloadOctree(vector<OctNode> &octree_vec, const int octree_size);
    void ConvertPointsToMorton(const int size, const int bits);
    void RadixSort(const int size, const Index mBits);
    void UniqueSorted(int &newSize);
	  void BuildBinaryRadixTree(size_t n, int mbits);
    void BinaryRadixToOctree(size_t n, int &octree_size);
};

