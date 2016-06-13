#pragma once
#ifdef __APPLE__
#include "OpenCL/cl.hpp"
#else
#include <CL/cl.hpp>
#endif
#include "clfw.hpp""
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "z_order.h"
extern "C" {
  #include "BrtNode.h"
  #include "BuildBRT.h"
  #include "OctNode.h"
  #include "BuildOctree.h"
  #include "ParallelAlgorithms.h"
}

using namespace std;
namespace KernelBox {
  int nextPow2(int num);
}