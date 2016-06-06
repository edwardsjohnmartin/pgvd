#pragma once
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include <CL/opencl.h>
#endif
#include "clfw.hpp"
#include "Buffers.h"
#include "Buffer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "z_order.h"
extern "C" {
  #include "BrtNode.h"
  #include "BuildBRT.h"
}

using namespace std;
namespace KernelBox {
  extern const vector<string> Files;
  extern unordered_map<string, cl_kernel> Kernels;
  extern Buffers buffers;
  
  bool IsProgramInitialized();
  bool IsInitialized();

  cl_int Initialize();
  cl_int BuildOpenCLProgram(const vector<string> Files);
  cl_int CreateKernels(cl_program program);

  bool isBufferUsable(shared_ptr<Buffer> buffer, size_t expectedSizeInBytes);
  cl_int createBuffer(shared_ptr<Buffer> &buffer, size_t size);
  int nextPow2(int num);
}