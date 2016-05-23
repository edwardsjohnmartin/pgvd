#include "KernelBox.h"
#include <algorithm>

void KernelBox::pointsToMorton(cl_mem input, cl_mem points, cl_int size, cl_int bits, size_t globalSize) {
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  
  error = clSetKernelArg(pointsToMortonKernel, 0, sizeof(cl_mem), &input);
  error = clSetKernelArg(pointsToMortonKernel, 1, sizeof(cl_mem), &points);
  error = clSetKernelArg(pointsToMortonKernel, 2, sizeof(cl_int), &size);
  error = clSetKernelArg(pointsToMortonKernel, 3, sizeof(cl_int), &bits);
  error = clEnqueueNDRangeKernel(queue, pointsToMortonKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox predication: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
};
void KernelBox::bitPredicate(cl_mem input, cl_mem predicate, Index &index, unsigned char compared, size_t globalSize){
	const size_t globalWorkSize[] = { globalSize, 0, 0 };
	cl_ulong i = index;
	unsigned char c = compared;
	error = clSetKernelArg(bitPredicateKernel, 0, sizeof (cl_mem), &input);
	error = clSetKernelArg(bitPredicateKernel, 1, sizeof (cl_mem), &predicate);
	error = clSetKernelArg(bitPredicateKernel, 2, sizeof(Index), &i);
	error = clSetKernelArg(bitPredicateKernel, 3, sizeof (unsigned char), &c);
	error = clEnqueueNDRangeKernel(queue, bitPredicateKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox predication: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
};
void KernelBox::uniquePredicate(cl_mem input, cl_mem predicate, size_t globalSize) {
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  error = clSetKernelArg(uniquePredicateKernel, 0, sizeof(cl_mem), &input);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox unique predication 1: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
  error = clSetKernelArg(uniquePredicateKernel, 1, sizeof(cl_mem), &predicate);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox unique predication 2: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
  error = clEnqueueNDRangeKernel(queue, uniquePredicateKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox unique predication: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
}
size_t KernelBox::getSteamScanWorkGroupSize(size_t globalSize) {
  using namespace std;
  size_t localSize;
  clGetKernelWorkGroupInfo(scanKernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &localSize, NULL);
  return min((int)(localSize), (int)globalSize);
}
void KernelBox::streamScan(cl_mem input, cl_mem intermediate, cl_mem intermediateCopy, cl_mem result, size_t globalSize){
  size_t localSize = getSteamScanWorkGroupSize(globalSize);
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
	const size_t localWorkSize[] = { localSize, 0, 0 };
  //std::cout << localSize << std::endl;

  int currentNumWorkgroups = (globalSize / localSize);
  clEnqueueCopyBuffer(queue, intermediateCopy, intermediate, 0, 0, sizeof(Index)* currentNumWorkgroups, 0, NULL, NULL);

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
  lastNumWorkgroups = currentNumWorkgroups;
};
void KernelBox::singleCompact(cl_mem inputBuffer, cl_mem resultBuffer, cl_mem PBuffer, cl_mem ABuffer, size_t globalSize) {
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  clSetKernelArg(singleCompactKernel, 0, sizeof(cl_mem), &inputBuffer);
  clSetKernelArg(singleCompactKernel, 1, sizeof(cl_mem), &resultBuffer);
  clSetKernelArg(singleCompactKernel, 2, sizeof(cl_mem), &PBuffer);
  clSetKernelArg(singleCompactKernel, 3, sizeof(cl_mem), &ABuffer);
  error = clEnqueueNDRangeKernel(queue, singleCompactKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
}
void KernelBox::doubleCompact(cl_mem inputBuffer, cl_mem resultBuffer, cl_mem resultBufferCopy, cl_mem LPBuffer, cl_mem LABuffer, size_t globalSize){
	const size_t globalWorkSize[] = { globalSize, 0, 0 };
	clSetKernelArg(doubleCompactKernel, 0, sizeof (cl_mem), &inputBuffer);
	clSetKernelArg(doubleCompactKernel, 1, sizeof (cl_mem), &resultBuffer);
	clSetKernelArg(doubleCompactKernel, 2, sizeof (cl_mem), &LPBuffer);
	clSetKernelArg(doubleCompactKernel, 3, sizeof (cl_mem), &LABuffer);
	clSetKernelArg(doubleCompactKernel, 4, sizeof (Index), &globalSize);

  BigUnsigned zero;
  initBlkBU(&zero, 0);
  //clEnqueueFillBuffer(queue, resultBuffer, &zero, sizeof(BigUnsigned), 0, sizeof(BigUnsigned)* (globalSize), 0, NULL, NULL);
  clEnqueueCopyBuffer(queue, resultBufferCopy, resultBuffer, 0, 0, sizeof(BigUnsigned) * globalSize, 0, NULL, NULL);
	error = clEnqueueNDRangeKernel(queue, doubleCompactKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
};
void KernelBox::buildBinaryRadixTree(cl_mem internalNodes, cl_mem leafNodes, cl_mem mpoints, cl_int mbits, cl_int n, size_t globalSize) {
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  clSetKernelArg(binaryRadixTreeKernel, 0, sizeof(cl_mem), &internalNodes);
  clSetKernelArg(binaryRadixTreeKernel, 1, sizeof(cl_mem), &leafNodes);
  clSetKernelArg(binaryRadixTreeKernel, 2, sizeof(cl_mem), &mpoints);
  clSetKernelArg(binaryRadixTreeKernel, 3, sizeof(cl_int), &mbits);
  clSetKernelArg(binaryRadixTreeKernel, 4, sizeof(cl_int), &n);

  error = clEnqueueNDRangeKernel(queue, binaryRadixTreeKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
}
void KernelBox::computeLocalSplits(cl_mem localSplits, cl_mem localSplitsCopy, cl_mem I, size_t size, size_t globalSize) {
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  clSetKernelArg(computeLocalSplitsKernel, 0, sizeof(cl_mem), &localSplits);
  clSetKernelArg(computeLocalSplitsKernel, 1, sizeof(cl_mem), &I);
  clSetKernelArg(computeLocalSplitsKernel, 2, sizeof(cl_mem), &size);

  clEnqueueCopyBuffer(queue, localSplitsCopy, localSplits, 0, 0, sizeof(unsigned int) * globalSize, 0, NULL, NULL);

  error = clEnqueueNDRangeKernel(queue, computeLocalSplitsKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
}
void KernelBox::brt2Octree(cl_mem I, cl_mem octree, cl_mem local_splits, cl_mem prefix_sums, cl_int n, size_t globalSize) {
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  clSetKernelArg(brt2OctreeKernel, 0, sizeof(cl_mem), &I);
  clSetKernelArg(brt2OctreeKernel, 1, sizeof(cl_mem), &octree);
  clSetKernelArg(brt2OctreeKernel, 2, sizeof(cl_mem), &local_splits);
  clSetKernelArg(brt2OctreeKernel, 3, sizeof(cl_mem), &prefix_sums);
  clSetKernelArg(brt2OctreeKernel, 4, sizeof(cl_int), &n);

  error = clEnqueueNDRangeKernel(queue, brt2OctreeKernel, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
}
void KernelBox::brt2Octree_init(cl_mem I, cl_mem octree, cl_mem local_splits, cl_mem prefix_sums, cl_int n, size_t globalSize) {
  const size_t globalWorkSize[] = { globalSize, 0, 0 };
  clSetKernelArg(brt2OctreeKernel_init, 0, sizeof(cl_mem), &I);
  clSetKernelArg(brt2OctreeKernel_init, 1, sizeof(cl_mem), &octree);
  clSetKernelArg(brt2OctreeKernel_init, 2, sizeof(cl_mem), &local_splits);
  clSetKernelArg(brt2OctreeKernel_init, 3, sizeof(cl_mem), &prefix_sums);
  clSetKernelArg(brt2OctreeKernel_init, 4, sizeof(cl_int), &n);

  error = clEnqueueNDRangeKernel(queue, brt2OctreeKernel_init, 1, 0, globalWorkSize, NULL, 0, nullptr, nullptr);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox double compaction: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
}


KernelBox::KernelBox(std::vector<std::string> fileNames, cl_context &context, cl_command_queue &_queue, cl_uint deviceIdCount, std::vector<cl_device_id> deviceIds)
{
	error = CL_SUCCESS;
	queue = _queue;
  device = deviceIds[0];
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
    std::cout << "KernelBox: Creating PointsToMortonKernel..." << std::endl;

  pointsToMortonKernel = clCreateKernel(program, "PointsToMortonKernel", &error);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }

  if (verbose)
		std::cout << "KernelBox: Creating BitPredicateKernel..." << std::endl;

  bitPredicateKernel = clCreateKernel(program, "BitPredicateKernel", &error);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}

  if (verbose)
    std::cout << "KernelBox: Creating Unique Predicate Kernel..." << std::endl;
  uniquePredicateKernel = clCreateKernel(program, "UniquePredicateKernel", &error);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }

	if (verbose)
		std::cout << "KernelBox: Creating StreamScanKernel..." << std::endl;
	scanKernel = clCreateKernel(program, "StreamScanKernel", &error);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
	if (verbose)
		std::cout << "KernelBox: Creating BUCompactKernel..." << std::endl;
	doubleCompactKernel = clCreateKernel(program, "BUCompactKernel", &error);
	if (error != CL_SUCCESS) {
		std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
		std::getchar();
		std::exit;
	}
  if (verbose)
    std::cout << "KernelBox: Creating BUSingleCompactKernel..." << std::endl;
  singleCompactKernel = clCreateKernel(program, "BUSingleCompactKernel", &error);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
  if (verbose)
    std::cout << "KernelBox: Creating BuildBinaryRadixTreeKernel..." << std::endl;
  binaryRadixTreeKernel = clCreateKernel(program, "BuildBinaryRadixTreeKernel", &error);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
  if (verbose)
    std::cout << "KernelBox: Creating ComputeLocalSplitsKernel..." << std::endl;
  computeLocalSplitsKernel = clCreateKernel(program, "ComputeLocalSplitsKernel", &error);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }

  if (verbose)
      std::cout << "KernelBox: Creating BRT2OctreeKernel_init..." << std::endl;
  brt2OctreeKernel_init = clCreateKernel(program, "BRT2OctreeKernel_init", &error);
  if (error != CL_SUCCESS) {
    std::cerr << "KernelBox: OpenCL call failed with error " << error << std::endl;
    std::getchar();
    std::exit;
  }
  if (verbose)
    std::cout << "KernelBox: Creating BRT2OctreeKernel..." << std::endl;
  brt2OctreeKernel = clCreateKernel(program, "BRT2OctreeKernel", &error);
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