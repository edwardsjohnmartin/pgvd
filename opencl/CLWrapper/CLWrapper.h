#pragma once
#include "..\Kernels\KernelBox.h"
#include <CL\cl.hpp>
#include <vector>
#include <iostream>
#include "..\..\C\BigUnsigned.h"

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
	std::vector<cl_platform_id> platformIds;
	std::vector<cl_device_id> deviceIds;
	KernelBox* kernelBox;

  std::vector<cl_mem> buffers;

	//Initializers
	void initPlatformIds();
	void initDevices();
	void initContext();
	void initCommandQueue();
	void initKernelBox();

	//Helper methods
	std::string getPlatformName(cl_platform_id id);
	std::string getDeviceName(cl_device_id id);
  void initRadixSortBuffers(std::vector<BigUnsigned> &input);
	void envokeRadixSortRoutine(const Index numBits);
	void CLWrapper::checkError(cl_int error);
public:
	bool verbose = true;
	size_t globalSize;
	size_t localSize;

	CLWrapper(size_t globalSize, size_t localSize);
	~CLWrapper();

	//Kernel Wrappers
	void RadixSort(std::vector<BigUnsigned> &input, const Index numBits);
};

