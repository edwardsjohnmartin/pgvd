#pragma once
#include <CL\cl.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include "..\..\C\BigUnsigned.h"

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
  cl_kernel predicateKernel;
  cl_program program;
	cl_kernel scanKernel;
	cl_kernel compactKernel;
	cl_command_queue queue;

	//Methods
	void KernelBox::initProgram(std::vector<std::string> fileNames, cl_context context, cl_uint deviceIdCount, std::vector<cl_device_id> deviceIds);
	void KernelBox::initKernels();

	//HelperMethods
	std::string loadKernel(const char* name);

public:
	bool verbose = true;
	KernelBox(std::vector<std::string> fileNames, cl_context &context, cl_command_queue &_queue, cl_uint deviceIdCount, std::vector<cl_device_id> deviceIds);
	~KernelBox();

	void predicate(cl_mem &input, cl_mem &predicate, Index &index, unsigned char compared, size_t globalSize, size_t localSize);
  void streamScan(cl_mem &input, cl_mem &intermediate, cl_mem &intermediateCopy, cl_mem &result, size_t globalSize, size_t localSize);
	void doubleCompact(cl_mem &inputBuffer, cl_mem &resultBuffer, cl_mem &LPBuffer, cl_mem &LABuffer, cl_mem &RABuffer, size_t globalSize, size_t localSize);
};

