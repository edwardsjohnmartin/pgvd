#include "KernelBox.h"

void KernelBox::predicate(cl_mem &input, cl_mem &predicate, Index &index, unsigned char compared, size_t globalSize, size_t localSize){
	const size_t globalWorkSize[] = { globalSize, 0, 0 };
	const size_t localWorkSize[] = { localSize, 0, 0 };
	cl_ulong i = index;
	unsigned char c = compared;
	error = clSetKernelArg(predicateKernel, 0, sizeof (cl_mem), &input);
	error = clSetKernelArg(predicateKernel, 1, sizeof (cl_mem), &predicate);
	error = clSetKernelArg(predicateKernel, 2, sizeof(unsigned short), &i);
	error = clSetKernelArg(predicateKernel, 3, sizeof (unsigned char), &c);
	error = clEnqueueNDRangeKernel(queue, predicateKernel, 1, 0, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox predication: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
};
void KernelBox::streamScan(cl_mem &input, cl_mem &intermediate, cl_mem &result, size_t globalSize, size_t localSize){
	const size_t globalWorkSize[] = { globalSize, 0, 0 };
	const size_t localWorkSize[] = { localSize, 0, 0 };

	Index* negativeOne = new Index(-1);
	clEnqueueFillBuffer(queue, intermediate, negativeOne, sizeof(Index), 0, sizeof(Index)* (globalSize / localSize), 0, NULL, NULL);

	clSetKernelArg(scanKernel, 0, sizeof (cl_mem), &input);
	clSetKernelArg(scanKernel, 1, sizeof (cl_mem), &result);
	clSetKernelArg(scanKernel, 2, sizeof (cl_mem), &intermediate);
	clSetKernelArg(scanKernel, 3, localSize * sizeof(Index), NULL);
	clSetKernelArg(scanKernel, 4, localSize * sizeof(Index), NULL);

	error = clEnqueueNDRangeKernel(queue, scanKernel, 1, 0, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox stream scan: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
	delete negativeOne;
};
void KernelBox::doubleCompact(cl_mem &inputBuffer, cl_mem &resultBuffer, cl_mem &LPBuffer, cl_mem &LABuffer, cl_mem &RABuffer, size_t globalSize, size_t localSize){
	const size_t globalWorkSize[] = { globalSize, 0, 0 };
	const size_t localWorkSize[] = { localSize, 0, 0 };
	clSetKernelArg(compactKernel, 0, sizeof (cl_mem), &inputBuffer);
	clSetKernelArg(compactKernel, 1, sizeof (cl_mem), &resultBuffer);
	clSetKernelArg(compactKernel, 2, sizeof (cl_mem), &LPBuffer);
	clSetKernelArg(compactKernel, 3, sizeof (cl_mem), &LABuffer);
	clSetKernelArg(compactKernel, 4, sizeof (cl_mem), &RABuffer);
	clSetKernelArg(compactKernel, 5, sizeof (unsigned short), &globalSize);
	error = clEnqueueNDRangeKernel(queue, compactKernel, 1, 0, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
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
cl_program KernelBox::CreateProgram(const std::string& source, cl_context context)
{
  size_t lengths[1] = { source.size() };
  const char* sources[1] = { source.data() };

  cl_int error = 0;
  cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }

  return program;
}
void KernelBox::initProgram(std::vector<std::string> fileNames, cl_context context, cl_uint deviceIdCount, std::vector<cl_device_id> deviceIds){
	using std::string;
  using namespace::std;
	if (verbose)
		std::cout << "KernelBox: Building kernel programs: ";
	
  string entireProgram;
	for (int i = 0; i < fileNames.size(); ++i)
    entireProgram += loadFile(fileNames[i].c_str());;
	
  program = CreateProgram(entireProgram, context);
  error = clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);// , program, deviceIds[0];

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
		//std::exit(1);
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

std::string KernelBox::loadFile(const char* name)
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