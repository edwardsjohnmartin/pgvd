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
#include "../Buffers/Buffers.h"
#include "../Buffers/Buffer.h"
#include "../../C/BrtNode.h"
#include "../../OctNode.h"
#include "../../opencl/vec.h"

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

	  //Initializers
	  void initPlatformIds();
	  void initDevices();
	  void initContext();
	  void initCommandQueue();
	  void initKernelBox();

	  //Private helper methods
	  string getPlatformName(cl_platform_id id);
	  string getDeviceName(cl_device_id id);
	  inline void initRadixSortBuffers();
	  void envokeRadixSortRoutine(const int size, const int bits, const Index mbits);
	  inline void initUniqueBuffers();
	  inline void initBrtBuffers();
    inline void initBRT2OctreeBuffers(size_t n);
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
    void RadixSort(const vector<intn>& points, const int bits, const Index mBits);
    size_t UniqueSorted();
	  void buildBrt(size_t n, int mbits);
    void BRT2Octree(size_t n, vector<OctNode> &octree_vec);
};

