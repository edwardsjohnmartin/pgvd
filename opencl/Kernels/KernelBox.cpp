#include "KernelBox.h"

void KernelBox::predicate(cl_mem &input, cl_mem &predicate, Index &index, unsigned char compared, size_t globalSize, size_t localSize){
	const size_t globalWorkSize[] = { globalSize, 0, 0 };
	const size_t localWorkSize[] = { localSize, 0, 0 };
	Index i = index;
	unsigned char c = compared;

	clSetKernelArg(predicateKernel, 0, sizeof (cl_mem), &input);
	clSetKernelArg(predicateKernel, 1, sizeof (cl_mem), &predicate);
	clSetKernelArg(predicateKernel, 2, sizeof (Index), &i);
	clSetKernelArg(predicateKernel, 3, sizeof (unsigned char), &c);
	error = clEnqueueNDRangeKernel(queue, predicateKernel, 1, 0, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
};
void KernelBox::streamScan(cl_mem &input, cl_mem &intermediate, cl_mem &intermediateCopy, cl_mem &result, size_t globalSize, size_t localSize){
	const size_t globalWorkSize[] = { globalSize, 0, 0 };
	const size_t localWorkSize[] = { localSize, 0, 0 };

  clEnqueueCopyBuffer(queue, intermediate, intermediateCopy, 0, 0, sizeof(Index)* (globalSize / localSize), 0, nullptr, nullptr);

	clSetKernelArg(scanKernel, 0, sizeof (cl_mem), &input);
	clSetKernelArg(scanKernel, 1, sizeof (cl_mem), &result);
	clSetKernelArg(scanKernel, 2, sizeof (cl_mem), &intermediateCopy);
	clSetKernelArg(scanKernel, 3, localSize * sizeof(Index), NULL);
	clSetKernelArg(scanKernel, 4, localSize * sizeof(Index), NULL);

	error = clEnqueueNDRangeKernel(queue, scanKernel, 1, 0, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
};
void KernelBox::doubleCompact(cl_mem &inputBuffer, cl_mem &resultBuffer, cl_mem &LPBuffer, cl_mem &LABuffer, cl_mem &RABuffer, size_t globalSize, size_t localSize){
	const size_t globalWorkSize[] = { globalSize, 0, 0 };
	const size_t localWorkSize[] = { localSize, 0, 0 };
	clSetKernelArg(compactKernel, 0, sizeof (cl_mem), &inputBuffer);
  clSetKernelArg(compactKernel, 1, sizeof (cl_mem), &resultBuffer);
	clSetKernelArg(compactKernel, 2, sizeof (cl_mem), &LPBuffer);
	clSetKernelArg(compactKernel, 3, sizeof (cl_mem), &LABuffer);
	clSetKernelArg(compactKernel, 4, sizeof (cl_mem), &RABuffer);
	clSetKernelArg(compactKernel, 5, sizeof (Index), &globalSize);
	error = clEnqueueNDRangeKernel(queue, compactKernel, 1, 0, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
};

KernelBox::KernelBox(std::vector<std::string> fileNames, cl_context &context, cl_command_queue &_queue, cl_uint deviceIdCount, std::vector<cl_device_id> deviceIds)
{
	error = CL_SUCCESS;
	queue = _queue;
	initProgram(fileNames, context, deviceIdCount, deviceIds);
	initKernels();
}
void KernelBox::initProgram(std::vector<std::string> fileNames, cl_context context, cl_uint deviceIdCount, std::vector<cl_device_id> deviceIds){
	using std::string;
	if (verbose)
		std::cout << "KernelBox: Building kernel programs: ";

	const char** sources = new const char*[fileNames.size()];
	string* ssources = new string[fileNames.size()];
	size_t* lengths = new size_t[fileNames.size()];

	int n = fileNames.size();
	n = 3;
	//std::vector<size_t> sourceLengths;
	for (int i = 0; i < n; ++i){
		std::string k = loadKernel(fileNames[i].c_str());
		ssources[i] = k;
		sources[i] = ssources[i].data();
		lengths[i] = k.size();
		//sourceLengths.push_back((size_t)k.size());
	  //std::cout << "File: " << (int)k.data()[k.length()] << std::endl;
	}
	std::cout << sources[2] << std::endl;

	program = clCreateProgramWithSource(context, n, sources, lengths, &error);
	error = clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);// , program, deviceIds[0];

	delete[] sources;
	delete[] ssources;
	delete[] lengths;

	if (error == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		getchar();
		std::exit(1);
	}
	else if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	} else if (verbose) {
		std::cout << "SUCCESS" << std::endl;
	}
}
void KernelBox::initKernels() {
	//Create Predication, Scan, & Compact kernels
	if (verbose)
		std::cout << "KernelBox: Creating Predicate Kernel..." << std::endl;

  predicateKernel = clCreateKernel(program, "Predicate", &error);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
	if (verbose)
		std::cout << "KernelBox: Creating StreamScan Kernel..." << std::endl;
	scanKernel = clCreateKernel(program, "StreamScan", &error);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
	if (verbose)
		std::cout << "KernelBox: Creating BUCompact Kernel..." << std::endl;
	compactKernel = clCreateKernel(program, "BUCompact", &error);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
}

std::string KernelBox::loadKernel(const char* name)
{
	std::ifstream in(name);
	std::string result(
		(std::istreambuf_iterator<char>(in)),
		std::istreambuf_iterator<char>());
	return result;
}

KernelBox::~KernelBox()
{
}