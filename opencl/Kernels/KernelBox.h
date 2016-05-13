#pragma once
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include <CL/cl.hpp>
#endif // __APPLE__

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include "../../C/BigUnsigned.h"

/*
  KernelBox. 
    -Builds openCL code
    -Provides kernel execution wrappers
    
  Nate B/V - bitinat2@isu.edu
*/
class KernelBox
{
private:
	//Variables
	cl_int error;
  cl_program program;
	cl_command_queue queue;
  cl_device_id device;
  cl_kernel bitPredicateKernel;
  cl_kernel uniquePredicateKernel;
	cl_kernel scanKernel;
  cl_kernel doubleCompactKernel;
  cl_kernel singleCompactKernel;
  cl_kernel binaryRadixTreeKernel;
  cl_kernel computeLocalSplitsKernel;
  cl_kernel brt2OctreeKernel_init;
  cl_kernel brt2OctreeKernel;

	//Methods
	void initProgram(std::vector<std::string> fileNames, cl_context context, cl_uint deviceIdCount, std::vector<cl_device_id> deviceIds);
  cl_program CreateProgram(const std::string& source, cl_context context);
	void initKernels();

	//HelperMethods
	std::string loadFile(const char* name);

public:
	bool verbose = true;
	KernelBox(std::vector<std::string> fileNames, cl_context &context, cl_command_queue &_queue, cl_uint deviceIdCount, std::vector<cl_device_id> deviceIds);
	~KernelBox();

	void bitPredicate(cl_mem input, cl_mem predicate, Index &index, unsigned char compared, size_t globalSize);
  void uniquePredicate(cl_mem input, cl_mem predicate, size_t globalSize);
	void streamScan(cl_mem input, cl_mem intermediate, cl_mem result, size_t globalSize);
  size_t getSteamScanWorkGroupSize(size_t globalSize);
  void doubleCompact(cl_mem inputBuffer, cl_mem resultBuffer, cl_mem LPBuffer, cl_mem LABuffer, cl_mem RABuffer, size_t globalSize);
  void singleCompact(cl_mem inputBuffer, cl_mem resultBuffer, cl_mem PBuffer, cl_mem ABuffer, size_t globalSize);
  void buildBinaryRadixTree(cl_mem internalNodes, cl_mem leafNodes, cl_mem mpoints, cl_int mbits, cl_int n, size_t globalSize);
  void computeLocalSplits(cl_mem localSplits, cl_mem I, size_t size, size_t globalSize);
  void brt2Octree(cl_mem I, cl_mem octree, cl_mem local_splits, cl_mem prefix_sums, cl_int n, size_t globalSize);
  void brt2Octree_init(cl_mem I, cl_mem octree, cl_mem local_splits, cl_mem prefix_sums, cl_int n, size_t globalSize);
};

