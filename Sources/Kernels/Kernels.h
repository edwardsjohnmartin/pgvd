/*
* Nate VM
* This file needs to be compiled by both OpenCL (SPIR-V or GPU compiler) and
*   a standard C++ compiler.
*
* All kernels in OpenCL preprocessor blocks are automatically compiled and
*  can be accessed using CLFW::Kernels[<kernel_function_name>] during runtime.
*/

#pragma region Include files
/* Host-Only Include files */
#ifndef OpenCL
// Had to move this here to compile on Mac
#include "GLUtilities/gl_utils.h"
#ifdef __APPLE__
#include "./cl.hpp"
#else
#include <CL/cl.hpp>
#endif
#include "BoundingBox/BoundingBox.h"
#include  "Options/options.h"
#include "./glm/gtc/matrix_transform.hpp"
#include "clfw.hpp"
#include "GLUtilities/Sketcher.h"
#include "Vector/vec.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
using namespace std;
#endif

/* Shared Include Files */
#include "CellResolution/Conflict.h"
#include "CellResolution/ConflictCellDetection.h"
#include "Octree/BuildOctree.h"  
#include "Octree/OctNode.h"  
#include "Quantize/Quantize.h"
#include "ZOrder/z_order.h"

#ifndef OpenCL
extern "C" {
#endif 

#include "BinaryRadixTree/BrtNode.h"
#include "BinaryRadixTree/BuildBRT.h"  
#include "Line/Line.h"
#include "OctreeResolution/Resln.h"
#include "ParallelAlgorithms/ParallelAlgorithms.h"  

#ifndef OpenCL
}
#endif 

#pragma endregion

#ifndef OpenCL
namespace Kernels {
#endif

	/* Definitions */
#pragma region Definitions
	/* These functions are called at the begining and ending of kernel execution.
	*  Currently, they aren't being used.
	*/
#define startBenchmark()
#define stopBenchmark()
#define benchmarking Options::debug
#pragma endregion

	/* Power Functions */
#pragma region Power Functions
#ifndef OpenCL
	inline int nextPow2(int num) {
		return max((int)pow(2, ceil(log((int)num) / log((int)2))), 8);
	}

	inline int isPow2(int num) {
		return ((num&(num - 1)) == 0);
	}
#endif
#pragma endregion

	/* Overloaded Operators*/
#pragma region Overloaded Operators
#ifndef OpenCL
	inline std::string buToString(BigUnsigned bu) {
		std::string representation = "";
		if (bu.len == 0)
		{
			representation += "[0]";
		}
		else {
			for (int i = bu.len; i > 0; --i) {
				representation += "[" + std::to_string(bu.blk[i - 1]) + "]";
			}
		}

		return representation;
	}

	inline std::string buToString(BigUnsigned bu, int len) {
		std::string representation = "";
		if (len == 0)
		{
			representation += "NULL";
		}
		else {
			//int shift = len%DIM;
			// len -= shift;
			for (int i = len - 1; i >= 0; --i) {
				representation += std::to_string(getBUBit(&bu, i));
			}
		}
		return representation;
	}

	inline std::ostream& operator<<(std::ostream& out, const BrtNode& node) {
		out << node.left << " " << node.left_leaf << " " << node.right_leaf << " " <<
			Kernels::buToString(node.lcp.bu) << " " << node.lcp.len << " " << node.parent;
		return out;
	}

	inline std::ostream& operator<<(std::ostream& out, const OctNode& node) {
		out << node.children[0] << " " << node.children[1] << " " << node.children[2] << " " << node.children[3] << " " <<
			node.leaf << " " << node.level<< " " << node.parent << " " << node.quadrant << " ";
		return out;
	}
#endif
#pragma endregion

	/* Reduction Kernels */
#pragma region Reduction Kernels

	// AddAll
#ifdef OpenCL
	__kernel
		void reduce(__global cl_int* buffer,
			__local cl_int* scratch,
			__const int length,
			__global cl_int* result)
	{
		//InitReduce
		int global_index = get_global_id(0);
		cl_int accumulator = 0;

		// Loop sequentially over chunks of input vector
		// improves Big O by Brent's Theorem.
		while (global_index < length) {
			accumulator += buffer[global_index];
			global_index += get_global_size(0);
		}

		// Perform parallel reduction
		int local_index = get_local_id(0);
		scratch[local_index] = accumulator;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
			if (local_index < offset)
				scratch[local_index] = scratch[local_index] + scratch[local_index + offset];
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if (local_index == 0) {
			result[get_group_id(0)] = scratch[0];
		}
	}
#else
	//Old, not optimized.
	inline cl_int AddAll(cl::Buffer &numbers, cl_uint& gpuSum, cl_int size) {
		startBenchmark();
		cl_int nextPowerOfTwo = nextPow2(size);
		if (nextPowerOfTwo != size) return CL_INVALID_ARG_SIZE;
		cl::Kernel &kernel = CLFW::Kernels["reduce"];
		cl::CommandQueue &queue = CLFW::DefaultQueue;

		//Each thread processes 2 items.
		int globalSize = size / 2;
		int suggestedLocal = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice);
		int localSize = std::min(globalSize, suggestedLocal);

		cl::Buffer reduceResult;
		cl_int resultSize = nextPow2(size / localSize);
		cl_int error = CLFW::get(reduceResult, "reduceResult", resultSize * sizeof(cl_uint));

		error |= kernel.setArg(0, numbers);
		error |= kernel.setArg(1, cl::__local(localSize * sizeof(cl_uint)));
		error |= kernel.setArg(2, nextPowerOfTwo);
		error |= kernel.setArg(3, reduceResult);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));

		//If multiple workgroups ran, we need to do a second level reduction.
		if (suggestedLocal <= globalSize) {
			error |= kernel.setArg(0, reduceResult);
			error |= kernel.setArg(1, cl::__local(localSize * sizeof(cl_uint)));
			error |= kernel.setArg(2, resultSize);
			error |= kernel.setArg(3, reduceResult);
			error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(localSize / 2), cl::NDRange(localSize / 2));
		}

		error |= queue.enqueueReadBuffer(reduceResult, CL_TRUE, 0, sizeof(cl_uint), &gpuSum);
		stopBenchmark();
		return error;
	}
#endif

	// Nvidia Reduce
#ifdef OpenCL
	/*
	* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
	*/

#define T int
#define blockSize 32
#define blockSize5 32
#define blockSize6 1024
#define nIsPow2 1
	/* Avoid using this kernel at all costs. This kernel is mainly for unit testing. Instead, use Reduce_s. */
	__kernel void oneThreadReduce(__global T *g_idata, __global T *g_odata, cl_int n) {
		if (get_global_id(0) == 0) {
			cl_int sum = 0;
			for (int i = 0; i < n; ++i) {
				sum += g_idata[i];
			}
			g_odata[0] = sum;
		}
	}
  
	/*
	This version uses n/2 threads --
	it performs the first level of reduction when reading from global memory
	*/
	__kernel void reduce3(__global T *g_idata, __global T *g_odata, cl_int n, __local T* sdata)
	{
		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		cl_int tid = get_local_id(0);
		cl_int i = get_group_id(0)*(get_local_size(0) * 2) + get_local_id(0);

		sdata[tid] = (i < n) ? g_idata[i] : 0;
		if (i + get_local_size(0) < n)
			sdata[tid] += g_idata[i + get_local_size(0)];

		barrier(CLK_LOCAL_MEM_FENCE);

		// do reduction in shared mem
		for (cl_int s = get_local_size(0) / 2; s > 0; s >>= 1)
		{
			if (tid < s)
			{
				sdata[tid] += sdata[tid + s];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		// write result for this block to global mem 
		if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
	}
	/*
	This version is completely unrolled.  It uses a template parameter to achieve
	optimal code for any (power of 2) number of threads.  This requires a switch
	statement in the host code to handle all the different thread block sizes at
	compile time.
	*/
	__kernel void reduce5(__global T *g_idata, __global T *g_odata, cl_int n, __local volatile T* sdata)
	{
		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		cl_int tid = get_local_id(0);
		cl_int i = get_group_id(0)*(get_local_size(0) * 2) + get_local_id(0);

		sdata[tid] = (i < n) ? g_idata[i] : 0;
		if (i + blockSize5 < n)
			sdata[tid] += g_idata[i + blockSize5];

		barrier(CLK_LOCAL_MEM_FENCE);

		// do reduction in shared mem
		if (blockSize5 >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } barrier(CLK_LOCAL_MEM_FENCE); }
		if (blockSize5 >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
		if (blockSize5 >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
		if (blockSize5 >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } barrier(CLK_LOCAL_MEM_FENCE); }

		if (tid < 32)
		{
			if (blockSize5 >= 64) { sdata[tid] += sdata[tid + 32]; }
			if (blockSize5 >= 32) { sdata[tid] += sdata[tid + 16]; }
			if (blockSize5 >= 16) { sdata[tid] += sdata[tid + 8]; }
			if (blockSize5 >= 8) { sdata[tid] += sdata[tid + 4]; }
			if (blockSize5 >= 4) { sdata[tid] += sdata[tid + 2]; }
			if (blockSize5 >= 2) { sdata[tid] += sdata[tid + 1]; }
		}

		// write result for this block to global mem 
		if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
	}

	/*
	This version adds multiple elements per thread sequentially.  This reduces the overall
	cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
	(Brent's Theorem optimization)
	*/
	__kernel void reduce6(__global T *g_idata, __global T *g_odata, cl_int n, __local volatile T* sdata)
	{
		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		cl_int tid = get_local_id(0);
		cl_int i = get_group_id(0)*(get_local_size(0) * 2) + get_local_id(0);
		cl_int gridSize = blockSize6 * 2 * get_num_groups(0);
		sdata[tid] = 0;

		// we reduce multiple elements per thread.  The number is determined by the 
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i < n)
		{
			sdata[tid] += g_idata[i];
			// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize6 < n)
				sdata[tid] += g_idata[i + blockSize6];
			i += gridSize;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// do reduction in shared mem
		if (blockSize6 >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } barrier(CLK_LOCAL_MEM_FENCE); }
		if (blockSize6 >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
		if (blockSize6 >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
		if (blockSize6 >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } barrier(CLK_LOCAL_MEM_FENCE); }

		if (tid < 32)
		{
			if (blockSize6 >= 64) { sdata[tid] += sdata[tid + 32]; }
			if (blockSize6 >= 32) { sdata[tid] += sdata[tid + 16]; }
			if (blockSize6 >= 16) { sdata[tid] += sdata[tid + 8]; }
			if (blockSize6 >= 8) { sdata[tid] += sdata[tid + 4]; }
			if (blockSize6 >= 4) { sdata[tid] += sdata[tid + 2]; }
			if (blockSize6 >= 2) { sdata[tid] += sdata[tid + 1]; }
		}

		// write result for this block to global mem 
		if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
	}

#else
	////////////////////////////////////////////////////////////////////////////////
	// Compute the number of threads and blocks to use for the given reduction kernel
	// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
	// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel 
	// 6, we observe the maximum specified number of blocks, because each thread in 
	// that kernel can process a variable number of elements.
	////////////////////////////////////////////////////////////////////////////////
	inline void getReductionNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
	{
		if (whichKernel < 3)
		{
			threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
			blocks = (n + threads - 1) / threads;
		}
		else
		{
			threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
			blocks = (n + (threads * 2 - 1)) / (threads * 2);
		}

		if (whichKernel == 6)
			blocks = std::min(maxBlocks, blocks);
	}

	inline cl_int Reduce_s(vector<cl_int> numbers_i, cl_int &result_o) {
		result_o = 0;
		for (int i = 0; i < numbers_i.size(); ++i) {
			result_o += numbers_i[i];
		}
		return CL_SUCCESS;
	}

	inline cl_int GeneralReduce(cl::Buffer inputBuffer, cl_int totalNumbers, string uniqueString, cl::Buffer &outputBuffer) {
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl_int error = 0;

		// Non-Nvidia platforms can't take advantage of Nvidia SIMD warp unrolling optimizations.
		// As a result, we default to the less efficient reduction number 3.
		cl::Kernel &kernel3 = CLFW::Kernels["reduce3"];
		error |= CLFW::get(outputBuffer, uniqueString + "reduceout", nextPow2(totalNumbers) * sizeof(cl_int));
    
		int maxThreads = std::min((int)kernel3.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice), 128);
		if (maxThreads == 1) {
			kernel3 = CLFW::Kernels["oneThreadReduce"];
			error |= kernel3.setArg(0, inputBuffer);
			error |= kernel3.setArg(1, outputBuffer);
			error |= kernel3.setArg(2, totalNumbers);
			error |= queue.enqueueNDRangeKernel(kernel3, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
			return error;
		}
		int maxBlocks = 64;

		int whichKernel = 3;
		int numBlocks = 0, finalNumBlocks = 0;
		int numThreads = 0, finalNumThreads = 0;
		int n = totalNumbers;
		getReductionNumBlocksAndThreads(whichKernel, totalNumbers, maxBlocks, maxThreads, numBlocks, numThreads);
		int s = numBlocks;

		//Initial reduction (per block reduction)
		error |= kernel3.setArg(0, inputBuffer);
		error |= kernel3.setArg(1, outputBuffer);
		error |= kernel3.setArg(2, n);
		error |= kernel3.setArg(3, cl::__local(numThreads * sizeof(cl_int)));

		int globalWorkSize = numBlocks * numThreads;
		int localWorkSize = numThreads;

		error |= queue.enqueueNDRangeKernel(kernel3, cl::NullRange,
			cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize));

		while (s > 1) {
			//Final reduction (adding each block's result together into one result)
			getReductionNumBlocksAndThreads(whichKernel, s, maxBlocks, maxThreads, finalNumBlocks, finalNumThreads);
			globalWorkSize = finalNumBlocks * finalNumThreads;
			localWorkSize = finalNumThreads;
			error |= kernel3.setArg(0, outputBuffer);
			error |= kernel3.setArg(1, outputBuffer);
			error |= kernel3.setArg(2, n);
			error |= kernel3.setArg(3, cl::__local(numThreads * sizeof(cl_int)));
			s = (s + (finalNumThreads * 2 - 1)) / (finalNumThreads * 2);
			error |= queue.enqueueNDRangeKernel(kernel3, cl::NullRange,
				cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize));
		}
		return error;
	}

	// Specifically calibrated to the gtx 1070, which recommends 1024 threads per block.
	// I'm getting about .250ms on 2^22 numbers, which is about 60GB/s
	inline cl_int NvidiaReduce() {
		if (CLFW::SelectedVendor != Vendor::Nvidia) return CL_INVALID_PLATFORM;

		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl_int error = 0;

		// Intel platforms can't take advantage of Nvidia SIMD warp unrolling optimizations.
		// As a result, we default to the less efficient reduction number 3.
		cl::Kernel &kernel6 = CLFW::Kernels["reduce6"];
		cl::Kernel &kernel5 = CLFW::Kernels["reduce5"];

		int size = 1 << 22; //will be a parameter
		int maxBlocks = 64;
		int maxThreads = min((int)kernel6.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice), 1024);

		cl::Buffer inputBuffer, outputBuffer;
		vector<cl_int> input(size);
		vector<cl_int> output(size);
		for (int i = 0; i < size; ++i) {
			input[i] = 1;
		}

		error |= CLFW::get(inputBuffer, "testnumbers", nextPow2(size) * sizeof(cl_int));
		error |= CLFW::get(outputBuffer, "resulttestnumbers", nextPow2(size) * sizeof(cl_int));
		error |= CLFW::Upload<cl_int>(input, inputBuffer);

		if (true || benchmarking) queue.finish();

		int whichKernel = 6;
		int numBlocks = 0, finalNumBlocks = 0;
		int numThreads = 0, finalNumThreads = 0;
		int n = size;
		getReductionNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);
		int s = numBlocks;

		//Initial reduction (per block reduction)
		kernel6.setArg(0, inputBuffer);
		kernel6.setArg(1, outputBuffer);
		kernel6.setArg(2, n);
		kernel6.setArg(3, cl::__local(numThreads * sizeof(cl_int)));

		int globalWorkSize = numBlocks * numThreads;
		int localWorkSize = numThreads;

		queue.enqueueNDRangeKernel(kernel6, cl::NullRange,
			cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize));

		whichKernel = 5;
		while (s > 1) {
			//Final reduction (adding each block's result together into one result)
			getReductionNumBlocksAndThreads(whichKernel, s, maxBlocks, maxThreads, finalNumBlocks, finalNumThreads);
			globalWorkSize = finalNumBlocks * finalNumThreads;
			localWorkSize = finalNumThreads;
			kernel5.setArg(0, outputBuffer);
			kernel5.setArg(1, outputBuffer);
			kernel5.setArg(2, n);
			kernel5.setArg(3, cl::__local(numThreads * sizeof(cl_int)));
			s = (s + (finalNumThreads * 2 - 1)) / (finalNumThreads * 2);
			queue.enqueueNDRangeKernel(kernel5, cl::NullRange,
				cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize));
		}

		if (true || benchmarking) {
			auto start = std::chrono::steady_clock::now();
			queue.finish();
			auto end = std::chrono::steady_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			std::cout << "Nvidia reduction time:" << elapsed.count() << " microseconds." << std::endl;
		}

		vector<cl_int> total(1);
		CLFW::Download<cl_int>(outputBuffer, 1, total);

		int cpusum = 0;
		for (int i = 0; i < size; ++i) {
			cpusum += input[i];
		}
		assert(cpusum == total[0]);

		return error;
	}

	inline cl_int Reduce_p(cl::Buffer numbers_i, cl_int totalNumbers, string uniqueString, cl::Buffer &result_o) {
		return GeneralReduce(numbers_i, totalNumbers, uniqueString, result_o);
		//if (CLFW::SelectedVendor == Vendor::Nvidia)
		//NvidiaReduce(numbers_i, totalNumbers, result_o);
		//else
	}

	inline cl_int ReduceTest() {
		/*if (CLFW::SelectedVendor != Vendor::Nvidia) {
			return GeneralReduce();
		}
		else
			return NvidiaReduce();*/
		return CL_SUCCESS;
	}
#endif

#pragma endregion

	/* Predication Kernels */
#pragma region Predication Kernels

	// Predicate Bit
#ifdef OpenCL
	__kernel void PredicateBitKernel(
		__global cl_int *inputBuffer,
		__global cl_int *predicateBuffer,
		cl_int index,
		cl_int comparedWith)
	{
		BitPredicate(inputBuffer, predicateBuffer, index, comparedWith, get_global_id(0));
	}
#else
	inline cl_int PredicateByBit_p(cl::Buffer &input, cl_int index, cl_int compared, cl_int totalElements, string uniqueString, cl::Buffer &predicate) {
		startBenchmark();
		cl_int roundSize = nextPow2(totalElements);
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["PredicateBitKernel"];

		cl_int error = CLFW::get(predicate, uniqueString + "predicateBit", sizeof(cl_int)* roundSize);

		error |= kernel->setArg(0, input);
		error |= kernel->setArg(1, predicate);
		error |= kernel->setArg(2, index);
		error |= kernel->setArg(3, compared);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(totalElements), cl::NullRange);
		stopBenchmark();
		return error;
	}
	inline cl_int PredicateByBit_s(vector<cl_int> numbers_i, cl_int index, cl_int compared, vector<cl_int> &predication_o) {
		startBenchmark();
		predication_o.resize(numbers_i.size());
		for (int i = 0; i < numbers_i.size(); ++i)
			BitPredicate(numbers_i.data(), predication_o.data(), index, compared, i);
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	// Predicate BU (BigUnsigned) Bit
#ifdef OpenCL
	__kernel void PredicateBUBitKernel(
		__global BigUnsigned *inputBuffer,
		__global cl_int *predicateBuffer,
		cl_int index,
		cl_int numIterations,
		cl_int shift,
		cl_int totalElements,
		cl_uchar comparedWith)
	{
#define ITERATIONS 4
#pragma unroll
		for (cl_int i = 0; i < ITERATIONS; ++i) {
			cl_int id = get_global_id(0) + i * shift;
			if (id < totalElements)
				BUBitPredicate(inputBuffer, predicateBuffer, index, comparedWith, id);
		}
#undef ITERATIONS
	}
#else
	inline cl_int PredicateBUByBit_p(cl::Buffer &input_i, cl_int index, cl_uchar compared, cl_int totalElements, string uniqueString, cl::Buffer &predicate_o) {
		startBenchmark();
		cl_int roundSize = nextPow2(totalElements);
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl::Kernel &kernel = CLFW::Kernels["PredicateBUBitKernel"];

		cl_int error = CLFW::get(predicate_o, uniqueString + "predicateBUBit", sizeof(cl_int)* (roundSize));
		cl_int numIterations = (roundSize > 128) ? 4 : 1;
		error |= kernel.setArg(0, input_i);
		error |= kernel.setArg(1, predicate_o);
		error |= kernel.setArg(2, index);
		error |= kernel.setArg(3, numIterations);
		error |= kernel.setArg(4, roundSize / numIterations);
		error |= kernel.setArg(5, totalElements);
		error |= kernel.setArg(6, compared);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(roundSize / numIterations), cl::NullRange);
		stopBenchmark();
		return error;
	};
	inline cl_int PredicateBUByBit_s(vector<BigUnsigned> numbers_i, cl_int index, cl_int compared, vector<cl_int> &predication_o) {
		startBenchmark();
		predication_o.resize(numbers_i.size());
		for (int i = 0; i < numbers_i.size(); ++i)
			BUBitPredicate(numbers_i.data(), predication_o.data(), index, compared, i);
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	// Predicate LCP
#ifdef OpenCL
	__kernel void PredicateLCPKernel(
		__global LCP *inputBuffer,
		__global cl_int *predicateBuffer,
		cl_int index,
		cl_int comparedWith,
		cl_int mbits)
	{
		LCPPredicate(inputBuffer, predicateBuffer, index, comparedWith, mbits, get_global_id(0));
	}
#else
	inline cl_int PredicateLCP_p(
		cl::Buffer &input_i,
		cl::Buffer &predicate_o,
		cl_int index,
		cl_int compared,
		cl_int totalElements,
		cl_int mbits)
	{
		startBenchmark();
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["PredicateLCPKernel"];
		int roundSize = nextPow2(totalElements);
		cl_int error = CLFW::get(predicate_o, "PredicateLCP", sizeof(cl_int)* (roundSize));

		error |= kernel->setArg(0, input_i);
		error |= kernel->setArg(1, predicate_o);
		error |= kernel->setArg(2, index);
		error |= kernel->setArg(3, compared);
		error |= kernel->setArg(4, mbits);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(totalElements), cl::NullRange);
		stopBenchmark();
		return error;
	};
#endif

	//Predicate Level
#ifdef OpenCL
	__kernel void PredicateLevelKernel(
		__global LCP *inputBuffer,
		__global cl_int *predicateBuffer,
		unsigned index,
		unsigned char comparedWith,
		int mbits)
	{
		LevelPredicate(inputBuffer, predicateBuffer, index, comparedWith, mbits, get_global_id(0));
	}
#else
	inline cl_int PredicateLevel(cl::Buffer &input, cl::Buffer &predicate, cl_int index, cl_int compared, cl_int totalElements, cl_int mbits) {
		startBenchmark();
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["PredicateLevelKernel"];
		int roundSize = nextPow2(totalElements);
		cl_int error = CLFW::get(predicate, "PredicateLevel", sizeof(cl_int)* (roundSize));

		error |= kernel->setArg(0, input);
		error |= kernel->setArg(1, predicate);
		error |= kernel->setArg(2, index);
		error |= kernel->setArg(3, compared);
		error |= kernel->setArg(4, mbits);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(totalElements), cl::NullRange);
		stopBenchmark();
		return error;
	};
#endif

	// Get Two Bit Mask
#ifdef OpenCL
	__kernel void GetTwoBitMaskKernel(
		__global BigUnsigned *inputBuffer,
		__global cl_int *masks,
		__local BigUnsigned *localBUBuffer,
		__local cl_int *localBoolBuffer,
		cl_int index,
		unsigned char compared
	)
	{
		const size_t gid = get_global_id(0);
		const size_t lid = get_local_id(0);

		localBUBuffer[lid] = inputBuffer[gid];
		GetTwoBitMask(localBUBuffer, localBoolBuffer, index, compared, lid);

		masks[gid * 4] = localBoolBuffer[lid * 4];
		masks[gid * 4 + 1] = localBoolBuffer[lid * 4 + 1];
		masks[gid * 4 + 2] = localBoolBuffer[lid * 4 + 2];
		masks[gid * 4 + 3] = localBoolBuffer[lid * 4 + 3];
	}
#else
	inline cl_int GetTwoBitMask_p(cl::Buffer &input, cl::Buffer &masks, cl_int index, unsigned char compared, cl_int size) {
		startBenchmark();
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["GetTwoBitMaskKernel"];
		int globalSize = nextPow2(size);
		int localSize = std::min((int)kernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice), globalSize);

		cl_int error = CLFW::get(masks, "masks", sizeof(cl_int) * (globalSize) * 4);

		error |= kernel->setArg(0, input);
		error |= kernel->setArg(1, masks);
		error |= kernel->setArg(2, cl::__local(localSize * sizeof(BigUnsigned)));
		error |= kernel->setArg(3, cl::__local(localSize * 4 * sizeof(cl_int)));
		error |= kernel->setArg(4, index);
		error |= kernel->setArg(5, compared);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
		stopBenchmark();
		return error;
	}
#endif
#ifndef OpenCL
	inline cl_int GetTwoBitMask_s(BigUnsigned* input, cl_int *masks, cl_int index, unsigned char compared, cl_int size) {
		startBenchmark();
		for (int i = 0; i < size; ++i) {
			GetTwoBitMask(input, masks, index, compared, i);
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	// Predicate Unique
#ifdef OpenCL
	__kernel void PredicateUniqueKernel(
		__global BigUnsigned *inputBuffer,
		__global cl_int *predicateBuffer)
	{
		BUUniquePredicate(inputBuffer, predicateBuffer, get_global_id(0));
	}
#else
	inline cl_int PredicateUnique_p(cl::Buffer &input, cl::Buffer &predicate, cl_int totalElements) {
		startBenchmark();
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["PredicateUniqueKernel"];

		cl_int error = kernel->setArg(0, input);
		error |= kernel->setArg(1, predicate);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(totalElements), cl::NullRange);

		stopBenchmark();
		return error;
	}
#endif

	/* Predicate Conflicts */
#ifdef OpenCL
	__kernel void PredicateConflictsKernel(
		__global Conflict *inputBuffer,
		__global cl_int *predicateBuffer 
		)
	{
		int gid = get_global_id(0);
		predicateBuffer[gid] = inputBuffer[gid].color == -2;
	}
#else
	inline cl_int PredicateConflicts_p(cl::Buffer &input, cl_int totalElements, string uniqueString, cl::Buffer &predicate) {
		startBenchmark();
		cl_int roundSize = nextPow2(totalElements);
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["PredicateConflictsKernel"];

		cl_int error = CLFW::get(predicate, uniqueString + "cPred", sizeof(cl_int)* roundSize);

		error |= kernel->setArg(0, input);
		error |= kernel->setArg(1, predicate);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(totalElements), cl::NullRange);
		stopBenchmark();
		return error;
	}
	inline cl_int PredicateConflicts_s(vector<Conflict> conflicts_i, vector<cl_int> &predication_o) {
		startBenchmark();
		predication_o.resize(conflicts_i.size());
		for (int i = 0; i < conflicts_i.size(); ++i)
			predication_o[i] = conflicts_i[i].color == -2;
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif
#pragma endregion

	/* Compaction Kernels */
#pragma region Compaction Kernels

	// Double Compact
#ifdef OpenCL
	__kernel void CompactKernel(
		__global cl_int *inputBuffer,
		__global cl_int *resultBuffer,
		__global cl_int *lPredicateBuffer,
		__global cl_int *leftBuffer,
		cl_int size)
	{
		Compact(inputBuffer, resultBuffer, lPredicateBuffer, leftBuffer, size, get_global_id(0));
	}
#else
	inline cl_int Compact_p(cl::Buffer &input_i, cl::Buffer &predicate_i, cl::Buffer &address_i, cl_int totalElements, cl::Buffer &result_i) {
		startBenchmark();
		cl_int error = 0;
		bool isOld;
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["CompactKernel"];
		cl::Buffer zeroBUBuffer;

		error |= CLFW::get(zeroBUBuffer, "zeroBuffer", sizeof(cl_int)*nextPow2(totalElements), isOld);
		if (!isOld) {
			error |= queue->enqueueFillBuffer<cl_int>(zeroBUBuffer, { 0 }, 0, nextPow2(totalElements) * sizeof(cl_int));
		}
		error |= queue->enqueueCopyBuffer(zeroBUBuffer, result_i, 0, 0, sizeof(cl_int) * totalElements);

		error |= kernel->setArg(0, input_i);
		error |= kernel->setArg(1, result_i);
		error |= kernel->setArg(2, predicate_i);
		error |= kernel->setArg(3, address_i);
		error |= kernel->setArg(4, totalElements);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(totalElements), cl::NullRange);
		stopBenchmark();
		return error;
	};
	inline cl_int Compact_s(vector<cl_int> &numbers_i, vector<cl_int> &predication, vector<cl_int> &addresses, vector<cl_int> &result_o) {
		cl_int size = numbers_i.size();
		result_o.resize(size);
		for (int i = 0; i < size; ++i) {
			Compact(numbers_i.data(), result_o.data(), predication.data(), addresses.data(), size, i);
		}
		return CL_SUCCESS;
	}
#endif

	// Single Compact
#ifdef OpenCL
	//Single Compaction
	__kernel void BUSingleCompactKernel(
		__global BigUnsigned *inputBuffer,
		__global BigUnsigned *resultBuffer,
		__global cl_int *predicateBuffer,
		__global cl_int *addressBuffer)
	{
		BUSingleCompact(inputBuffer, resultBuffer, predicateBuffer, addressBuffer, get_global_id(0));
	}
#else
	inline cl_int BUSingleCompact(cl::Buffer &input, cl::Buffer &result, cl::Memory &predicate, cl::Buffer &address, cl_int totalElements) {
		startBenchmark();
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["BUSingleCompactKernel"];

		cl_int error = kernel->setArg(0, input);
		error |= kernel->setArg(1, result);
		error |= kernel->setArg(2, predicate);
		error |= kernel->setArg(3, address);

		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(totalElements), cl::NullRange);
		stopBenchmark();
		return error;
	}
#endif

	// Leaf Double Compact
#ifdef OpenCL
	__kernel void LeafDoubleCompactKernel(
		__global Leaf *inputBuffer,
		__global Leaf *resultBuffer,
		__global cl_int *predicateBuffer,
		__global cl_int *addressBuffer,
		cl_int size)
	{
		LeafDoubleCompact(inputBuffer, resultBuffer, predicateBuffer, addressBuffer, size, get_global_id(0));
	}
#else
	inline cl_int LeafDoubleCompact(cl::Buffer &input_i, cl::Buffer &result_i, cl::Memory &predicate_i, cl::Buffer &address_i, cl_int globalSize) {
		//startBenchmark();
		cl_int error = 0;
		bool isOld;
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["LeafDoubleCompactKernel"];
		cl::Buffer zeroLeafBuffer;

		error |= CLFW::get(zeroLeafBuffer, "zeroLeafBuffer", sizeof(Leaf)*globalSize, isOld);
		if (!isOld) {
			Leaf zeroLeaf;
			zeroLeaf.parent = -42;
			zeroLeaf.quadrant = -42;
			error |= queue->enqueueFillBuffer<Leaf>(zeroLeafBuffer, zeroLeaf, 0, globalSize * sizeof(Leaf));
		}
		error |= queue->enqueueCopyBuffer(zeroLeafBuffer, result_i, 0, 0, sizeof(Leaf) * globalSize);

		error |= kernel->setArg(0, input_i);
		error |= kernel->setArg(1, result_i);
		error |= kernel->setArg(2, predicate_i);
		error |= kernel->setArg(3, address_i);
		error |= kernel->setArg(4, globalSize);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
		//stopBenchmark();
		return error;
	}
#endif

	// BU Double Compact
#ifdef OpenCL
	//Double Compaction
	__kernel void BUCompactKernel(
		__global BigUnsigned *inputBuffer,
		__global BigUnsigned *resultBuffer,
		__global cl_int *lPredicateBuffer,
		__global cl_int *leftBuffer,
		cl_int size,
		cl_int shift)
	{
#define ITERATIONS 4
#pragma unroll
		for (int i = 0; i < ITERATIONS; ++i) {
			int id = get_global_id(0) + i * shift;
			if (id < size) {
				BUCompact(inputBuffer, resultBuffer, lPredicateBuffer, leftBuffer, size, id);
			}
		}
#undef ITERATIONS
	}
#else
	inline cl_int BUCompact_p(cl::Buffer &input, cl_int totalElements, cl::Buffer &predicate, cl::Buffer &address, cl::Buffer &result) {
		//startBenchmark();
		cl_int globalSize = nextPow2(totalElements);
		cl_int error = 0;
		bool isOld;
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["BUCompactKernel"];
		cl::Buffer zeroBUBuffer;

		error |= CLFW::get(zeroBUBuffer, "zeroBUBuffer", sizeof(BigUnsigned)*globalSize, isOld);
		if (!isOld) {
			BigUnsigned zero;
			initBlkBU(&zero, 0);
			error |= queue->enqueueFillBuffer<BigUnsigned>(zeroBUBuffer, { zero }, 0, globalSize * sizeof(BigUnsigned));
		}
		error |= queue->enqueueCopyBuffer(zeroBUBuffer, result, 0, 0, sizeof(BigUnsigned) * totalElements);

		int numIterations = (globalSize > 128) ? 4 : 1;
		int localSize = min(32, globalSize / numIterations);
		error |= kernel->setArg(0, input);
		error |= kernel->setArg(1, result);
		error |= kernel->setArg(2, predicate);
		error |= kernel->setArg(3, address);
		error |= kernel->setArg(4, totalElements);
		error |= kernel->setArg(5, globalSize / numIterations);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize / numIterations), cl::NDRange(localSize));
		//stopBenchmark();
		return error;
	};
	inline cl_int BUCompact_s(vector<BigUnsigned> &numbers_i, vector<cl_int> &predication, vector<cl_int> &addresses, vector<BigUnsigned> &result_o) {
		cl_int size = numbers_i.size();
		result_o.resize(size);
		for (int i = 0; i < size; ++i) {
			BUCompact(numbers_i.data(), result_o.data(), predication.data(), addresses.data(), size, i);
		}
		return CL_SUCCESS;
	}
#endif

	// LCP Facet Double Compact
#ifdef OpenCL
	//Double Compaction
	__kernel void LCPFacetCompactKernel(
		__global LCP *inputLCPBuffer,
		__global LCP *resultLCPBuffer,
		__global cl_int *inputIndexBuffer,
		__global cl_int *resultIndexBuffer,
		__global cl_int *lPredicateBuffer,
		__global cl_int *leftBuffer,
		cl_int size)
	{
		LCPFacetCompact(inputLCPBuffer, inputIndexBuffer, resultLCPBuffer, resultIndexBuffer, lPredicateBuffer, leftBuffer, size, get_global_id(0));
	}
#else
	inline cl_int LCPFacetDoubleCompact(
		cl::Buffer &inputLCPs_i,
		cl::Buffer &inputIndices_i,
		cl::Buffer &resultLCPs_i,
		cl::Buffer &resultIndices_i,
		cl::Buffer &predicate_i,
		cl::Buffer &address_i,
		cl_int globalSize)
	{
		startBenchmark();
		cl_int error = 0;
		bool isOld;
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["LCPFacetCompactKernel"];
		cl::Buffer zeroLCPBuffer;
		cl::Buffer zeroIndexBuffer;
		cl_int roundSize = nextPow2(globalSize);
		error |= CLFW::get(zeroLCPBuffer, "zeroLCPBuffer", sizeof(LCP)*roundSize, isOld);
		error |= CLFW::get(zeroIndexBuffer, "zeroIndexBuffer", sizeof(cl_int)*roundSize, isOld);


		if (!isOld) {
			LCP zero;
			initBlkBU(&zero.bu, 0);
			zero.len = -1;
			error |= queue->enqueueFillBuffer<LCP>(zeroLCPBuffer, { zero }, 0, roundSize * sizeof(LCP));
			error |= queue->enqueueFillBuffer<cl_int>(zeroIndexBuffer, { 0 }, 0, roundSize * sizeof(cl_int));
		}
		error |= queue->enqueueCopyBuffer(zeroLCPBuffer, resultLCPs_i, 0, 0, sizeof(LCP) * globalSize);
		error |= queue->enqueueCopyBuffer(zeroIndexBuffer, resultIndices_i, 0, 0, sizeof(cl_int) * globalSize);


		error |= kernel->setArg(0, inputLCPs_i);
		error |= kernel->setArg(1, resultLCPs_i);
		error |= kernel->setArg(2, inputIndices_i);
		error |= kernel->setArg(3, resultIndices_i);
		error |= kernel->setArg(4, predicate_i);
		error |= kernel->setArg(5, address_i);
		error |= kernel->setArg(6, globalSize);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
		stopBenchmark();
		return error;
	};
#endif

	// Conflcit Double Compaction
#ifdef OpenCL
	__kernel void CompactConflictsKernel(
		__global Conflict *inputBuffer,
		__global Conflict *resultBuffer,
		__global cl_int *lPredicateBuffer,
		__global cl_int *leftBuffer,
		cl_int size)
	{
		CompactConflicts(inputBuffer, resultBuffer, lPredicateBuffer, leftBuffer, size, get_global_id(0));
	}
#else
	inline cl_int CompactConflicts_p(cl::Buffer &input_i, cl::Buffer &predicate_i, cl::Buffer &address_i, cl_int totalElements, cl::Buffer &result_io) {
		startBenchmark();
		cl_int error = 0;
		bool isOld;
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["CompactConflictsKernel"];
		cl::Buffer zeroBUBuffer;

		error |= kernel->setArg(0, input_i);
		error |= kernel->setArg(1, result_io);
		error |= kernel->setArg(2, predicate_i);
		error |= kernel->setArg(3, address_i);
		error |= kernel->setArg(4, totalElements);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(totalElements), cl::NullRange);
		stopBenchmark();
		return error;
	};
	inline cl_int CompactConflicts_s(vector<Conflict> &conflicts_i, vector<cl_int> &predication_i, vector<cl_int> &addresses_i, vector<Conflict> &result_o) {
		cl_int size = conflicts_i.size();
		result_o.resize(size);
		for (int i = 0; i < size; ++i) {
			CompactConflicts(conflicts_i.data(), result_o.data(), predication_i.data(), addresses_i.data(), size, i);
		}
		return CL_SUCCESS;
	}
#endif

	// Octnode Double Compact
#ifdef OpenCL
	//Needs implementing
#else
	inline cl_int OctnodeDoubleCompact(cl::Buffer &input_i, cl::Buffer &result_i, cl::Buffer &predicate_i, cl::Buffer &address_i, cl_int globalSize) {
		//startBenchmark();
		cl_int error = 0;
		bool isOld;
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["OctnodeCompactKernel"];
		cl::Buffer zeroOctnodeBuffer;

		error |= CLFW::get(zeroOctnodeBuffer, "zeroOctnodeBuffer", sizeof(OctNode)*globalSize, isOld);
		if (!isOld) {
			OctNode zero;
			zero.children[0] = zero.children[1] = zero.children[2] = zero.children[3] = -1;
			zero.leaf = -1;
			zero.parent = -1;
			zero.level = -1;
			zero.quadrant = -1;
			error |= queue->enqueueFillBuffer<OctNode>(zeroOctnodeBuffer, { zero }, 0, globalSize * sizeof(OctNode));
		}
		error |= queue->enqueueCopyBuffer(zeroOctnodeBuffer, result_i, 0, 0, sizeof(OctNode) * globalSize);

		int numIterations = 4;
		error |= kernel->setArg(0, input_i);
		error |= kernel->setArg(1, result_i);
		error |= kernel->setArg(2, predicate_i);
		error |= kernel->setArg(3, address_i);
		error |= kernel->setArg(4, globalSize);
		error |= kernel->setArg(5, globalSize / numIterations);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(globalSize / numIterations), cl::NDRange(32));
		//stopBenchmark();
		return error;
	};
#endif

#pragma endregion

	/* Scan Kernels */
#pragma region Scan/Prefix Sum Kernels

	// Get Four Way Prefix Sum
#ifdef OpenCL
	//Needs implementing
#else
	inline cl_int GetFourWayPrefixSum_p(cl::Buffer &input, cl::Buffer &masks, cl_int index, unsigned char compared, cl_int size)
	{
		return -1;
	}
#endif
#ifndef OpenCL
	inline cl_int GetFourWayPrefixSum_s(BigUnsigned* input, cl_int *fourWayPrefix, cl_int index, unsigned char compared, cl_int size)
	{
		startBenchmark();
		vector<cl_int> masks(size * 4);

		for (int i = 0; i < size; ++i) {
			GetTwoBitMask(input, masks.data(), index, compared, i);
		}

		fourWayPrefix[0] = fourWayPrefix[1] = fourWayPrefix[2] = fourWayPrefix[3] = 0;
		for (int i = 1; i < size; ++i)
		{
			fourWayPrefix[i * 4] = masks[(i - 1) * 4] + fourWayPrefix[(i - 1) * 4];
			fourWayPrefix[i * 4 + 1] = masks[(i - 1) * 4 + 1] + fourWayPrefix[(i - 1) * 4 + 1];
			fourWayPrefix[i * 4 + 2] = masks[(i - 1) * 4 + 2] + fourWayPrefix[(i - 1) * 4 + 2];
			fourWayPrefix[i * 4 + 3] = masks[(i - 1) * 4 + 3] + fourWayPrefix[(i - 1) * 4 + 3];
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	// Stream Scan
#ifdef OpenCL
	__kernel void StreamScanKernel(
		__global cl_int* buffer,
		__global cl_int* result,
		__global volatile cl_int* I,
		__local cl_int* localBuffer,
		__local cl_int* scratch,
		cl_int totalElements)
	{
		const cl_int gid = get_global_id(0);
		const cl_int lid = get_local_id(0);
		const cl_int wid = get_group_id(0);
		const cl_int ls = get_local_size(0);
		cl_int sum = 0;
		if (gid < totalElements)
			localBuffer[lid] = scratch[lid] = buffer[gid];
		else
			localBuffer[lid] = scratch[lid] = 0;
		barrier(CLK_LOCAL_MEM_FENCE);

		//Reduce
		for (cl_int offset = ls / 2; offset > 0; offset >>= 1) {
			if (lid < offset)
				scratch[lid] = scratch[lid + offset] + scratch[lid];
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		//Do Adjacent sync
		if (lid == 0 && gid != 0) {
			while (I[wid - 1] == -1);
			I[wid] = I[wid - 1] + scratch[0];
		}
		if (gid == 0) I[0] = scratch[0];
		barrier(CLK_LOCAL_MEM_FENCE);

		//Local Scan
		scratch[lid] = localBuffer[lid];
		for (cl_int i = 1; i < ls; i <<= 1) {
			if (lid > (i - 1))
				scratch[lid] = localBuffer[lid] + localBuffer[lid - i];
			else
				scratch[lid] = localBuffer[lid];
			__local cl_int *tmp = scratch;
			scratch = localBuffer;
			localBuffer = tmp;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if (gid < totalElements) {
			sum = localBuffer[lid];
			if (wid != 0) sum += I[wid - 1];
			result[gid] = sum;
		}
	}
#else
	inline cl_int StreamScan_p(
		cl::Buffer &input_i, const cl_int totalElements,
		string uniqueString, cl::Buffer &result_i) {
		int globalSize = nextPow2(totalElements);
		startBenchmark();
		cl_int error = 0;
		bool isOld;
		cl::Kernel &kernel = CLFW::Kernels["StreamScanKernel"];
		cl::CommandQueue &queue = CLFW::DefaultQueue;

		/* Determine the number of groups required. */
		int wgSize = std::min((int)kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice), globalSize);
		int localSize = std::min(wgSize, globalSize);
		int numGroups = (globalSize / wgSize); //+1

		/* Each workgroup gets a spot in the intermediate buffer. */
		cl::Buffer intermediate, intermediateCopy;
		error |= CLFW::get(intermediate, uniqueString + "I", sizeof(cl_int) * numGroups);
		error |= CLFW::get(intermediateCopy, uniqueString + "Icopy",
			sizeof(cl_int) * numGroups, isOld);
		if (!isOld)
			error |= queue.enqueueFillBuffer<cl_int>(
				intermediateCopy, { -1 }, 0, sizeof(cl_int) * numGroups);
		error |= queue.enqueueCopyBuffer(intermediateCopy, intermediate, 0, 0,
			sizeof(cl_int) * numGroups);

		/* Call the kernel */
		error |= kernel.setArg(0, input_i);
		error |= kernel.setArg(1, result_i);
		error |= kernel.setArg(2, intermediate);
		error |= kernel.setArg(3, cl::__local(localSize * sizeof(cl_int)));
		error |= kernel.setArg(4, cl::__local(localSize * sizeof(cl_int)));
		error |= kernel.setArg(5, totalElements);
		error |= queue.enqueueNDRangeKernel(
			kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));
		stopBenchmark();
		return error;
	}
#endif
#ifndef OpenCL
	inline cl_int StreamScan_s(vector<cl_int> &buffer, vector<cl_int> &result) {
		startBenchmark();
		cl_int size = buffer.size();
		cl_int nextPowerOfTwo = (int)pow(2, ceil(log(size) / log(2)));
		cl_int intermediate = -1;
		cl_int* localBuffer;
		cl_int* scratch;
		cl_int sum = 0;

		localBuffer = (cl_int*)malloc(sizeof(cl_int)* nextPowerOfTwo);
		scratch = (cl_int*)malloc(sizeof(cl_int)* nextPowerOfTwo);
		//INIT
		for (cl_int i = 0; i < size; i++)
			localBuffer[i] = scratch[i] = buffer[i];
		for (cl_int i = size; i < nextPowerOfTwo; ++i)
			localBuffer[i] = scratch[i] = 0;

		//Add not necessary with only one workgroup.
		//Adjacent sync not necessary with only one workgroup.

		//SCAN
		for (cl_int i = 1; i < nextPowerOfTwo; i <<= 1) {
			for (cl_int j = 0; j < nextPowerOfTwo; ++j) {
				if (j > (i - 1))
					scratch[j] = localBuffer[j] + localBuffer[j - i];
				else
					scratch[j] = localBuffer[j];
			}
			cl_int *tmp = scratch;
			scratch = localBuffer;
			localBuffer = tmp;
		}
		for (cl_int i = 0; i < size; ++i) {
			result[i] = localBuffer[i];
		}
		free(localBuffer);
		free(scratch);

		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

#ifdef OpenCL
#define scanBlockSize 1024
	__kernel void NvidiaStreamScanKernel(
		__global cl_int* buffer,
		__global cl_int* result,
		__global volatile int* I,
		__local volatile cl_int* scratch2,
		__local volatile cl_int* scratch)
	{
		const size_t gid = get_global_id(0);
		const size_t lid = get_local_id(0);
		const size_t wid = get_group_id(0);
		const size_t ls = get_local_size(0);
		int sum = 0;
		//FIX THIS
		cl_int n = 1 << 22;


		cl_int tid = get_local_id(0);
		cl_int i = get_group_id(0)*(get_local_size(0) * 2) + get_local_id(0);

		scratch[tid] = (i < n) ? buffer[i] : 0;
		if (tid < 512) {
			// if (i + scanBlockSize < n) 
			//     scratch[tid] += buffer[i+scanBlockSize];  

			// barrier(CLK_LOCAL_MEM_FENCE);

			// // do reduction in shared mem
			// if (scanBlockSize >= 1024) { if (tid < 512) { scratch[tid] += scratch[tid + 512]; } barrier(CLK_LOCAL_MEM_FENCE); }
			// if (scanBlockSize >= 512 ) { if (tid < 256) { scratch[tid] += scratch[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
			// if (scanBlockSize >= 256 ) { if (tid < 128) { scratch[tid] += scratch[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
			// if (scanBlockSize >= 128 ) { if (tid <  64) { scratch[tid] += scratch[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }

			// if (tid < 32)
			// {
			//     if (scanBlockSize >=  64) { scratch[tid] += scratch[tid + 32]; }
			//     if (scanBlockSize >=  32) { scratch[tid] += scratch[tid + 16]; }
			//     if (scanBlockSize >=  16) { scratch[tid] += scratch[tid +  8]; }
			//     if (scanBlockSize >=   8) { scratch[tid] += scratch[tid +  4]; }
			//     if (scanBlockSize >=   4) { scratch[tid] += scratch[tid +  2]; }
			//     if (scanBlockSize >=   2) { scratch[tid] += scratch[tid +  1]; }
			// }
		}
		else {
			scratch[lid] = scratch2[lid];
			for (cl_int i = 1; i < ls; i <<= 1) {
				if (lid > (i - 1))
					scratch[lid] = scratch2[lid] + scratch2[lid - i];
				else
					scratch[lid] = scratch2[lid];
				volatile __local cl_int *tmp = scratch;
				scratch = scratch2;
				scratch2 = tmp;
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			sum = scratch2[lid];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		// if (lid == 0)
		//   result[gid] = scratch[lid];
		// else
		//   result[gid] = 0;
		// int sum = 0;  
		// StreamScan_Init(buffer, scratch2, scratch, gid, lid);
		// barrier(CLK_LOCAL_MEM_FENCE);

		// #pragma unroll 1
		// for (int offset = ls / 2; offset > 32; offset >>= 1) {
		//   AddAll(scratch, lid, offset);
		//   barrier(CLK_LOCAL_MEM_FENCE);
		// }
		// if (lid < 32)
		// {
		//     if (scanBlockSize >=  64) { scratch[lid] += scratch[lid + 32]; }
		//     if (scanBlockSize >=  32) { scratch[lid] += scratch[lid + 16]; }
		//     if (scanBlockSize >=  16) { scratch[lid] += scratch[lid +  8]; }
		//     // if (scanBlockSize >=   8) { scratch[lid] += scratch[lid +  4]; }
		//     // barrier(CLK_LOCAL_MEM_FENCE);
		//     // if (scanBlockSize >=   4) { scratch[lid] += scratch[lid +  2]; }
		//     // barrier(CLK_LOCAL_MEM_FENCE);
		//     // if (scanBlockSize >=   2) { scratch[lid] += scratch[lid +  1]; }
		//     // barrier(CLK_LOCAL_MEM_FENCE);

		// }


		//ADJACENT SYNCRONIZATION
		if (lid == 0 && gid != 0) {
			while (I[wid - 1] == -1);
			I[wid] = I[wid - 1] + scratch[0];
		}
		if (gid == 0) I[0] = scratch[0];

		barrier(CLK_LOCAL_MEM_FENCE);

		result[gid] = scratch2[lid];//scratch[0];



																//   if (wid != 0) sum += I[wid - 1];
																//   result[gid] = sum; 
	}
#else
	inline cl_int StreamScanTest() {
		cl_int error = 0;
		cl::Kernel &kernel = CLFW::Kernels["NvidiaStreamScanKernel"];
		cl::CommandQueue *queue = &CLFW::DefaultQueue;

		int size = 1 << 22; //will be a parameter

		cl::Buffer inputBuffer, outputBuffer;
		vector<cl_int> input(size);
		vector<cl_int> output(size);
		for (int i = 0; i < size; ++i) {
			input[i] = 1;
		}

		error |= CLFW::get(inputBuffer, "scanIn", nextPow2(size) * sizeof(cl_int));
		error |= CLFW::get(outputBuffer, "scanout", nextPow2(size) * sizeof(cl_int));
		error |= CLFW::Upload<cl_int>(input, inputBuffer);
		CLFW::DefaultQueue.finish();

		const int wgSize = (int)kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice);
		const int globalSize = size;
		const int localSize = std::min(wgSize, globalSize);
		int currentNumWorkgroups = (globalSize / localSize) + 1;

		bool isOld;
		cl::Buffer intermediate, intermediateCopy;
		error |= CLFW::get(intermediate, "StreamScanTestIntermediate",
			sizeof(cl_int) * currentNumWorkgroups);
		error |= CLFW::get(intermediateCopy, "StreamScanTestIntermediateCopy",
			sizeof(cl_int) * currentNumWorkgroups, isOld);

		if (!isOld) {
			error |= queue->enqueueFillBuffer<cl_int>(
				intermediateCopy, { -1 }, 0, sizeof(cl_int) * currentNumWorkgroups);
			assert_cl_error(error);
		}
		error |= queue->enqueueCopyBuffer(
			intermediateCopy, intermediate, 0, 0,
			sizeof(cl_int) * currentNumWorkgroups);

		if (true || benchmarking) queue->finish();
		error |= kernel.setArg(0, inputBuffer);
		error |= kernel.setArg(1, outputBuffer);
		error |= kernel.setArg(2, intermediate);
		error |= kernel.setArg(3, cl::__local(localSize * sizeof(cl_int)));
		error |= kernel.setArg(4, cl::__local(localSize * sizeof(cl_int)));
		error |= queue->enqueueNDRangeKernel(
			kernel, cl::NullRange, cl::NDRange(nextPow2(globalSize)), cl::NDRange(localSize));

		if (true || benchmarking) {
			auto start = std::chrono::steady_clock::now();
			CLFW::DefaultQueue.finish();
			auto end = std::chrono::steady_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			std::cout << "streamscan time:" << elapsed.count() << " microseconds." << std::endl;
		}

		vector<cl_int> cpuOutput(size);

		cpuOutput[0] = input[0];
		for (int i = 1; i < size; ++i) {
			cpuOutput[i] = cpuOutput[i - 1] + input[i];
		}

		CLFW::Download<cl_int>(outputBuffer, size, output);
		for (int i = 0; i < size; ++i) {
			assert(cpuOutput[i] == output[i]);
		}

		//needs testing
		return error;
	}
#endif
#pragma endregion

	/* Sort Routines */
#pragma region Sort Routines

	// Radix Sort BigUnsigned
#ifndef OpenCL
	//Approx 16% of total build
	inline cl_int RadixSortBigUnsigned_p(
		cl::Buffer &numbers_io,
		cl_int totalPoints,
		cl_int mbits,
		string uniqueString
	) {
		startBenchmark();

		cl_int error = 0;
		const cl_int globalSize = nextPow2(totalPoints);
		cl::Buffer predicate, address, temp, swap;
		error |= CLFW::get(address, uniqueString + "BUAddress", sizeof(cl_int)*(globalSize));
		error |= CLFW::get(temp, uniqueString + "BUTemp", sizeof(BigUnsigned)*globalSize);

		if (error != CL_SUCCESS)
			return error;

		//For each bit
		for (cl_int index = 0; index < mbits; index++) { //each loop 450micro->700micro
			//Predicate the 0's and 1's
			error |= PredicateBUByBit_p(numbers_io, index, 0, totalPoints, "rdxsrtbu", predicate); //130micro

			//Scan the predication buffers.
			error |= StreamScan_p(predicate, totalPoints, uniqueString + "RSBUI", address); //300->600micro

			//Compacting
			error |= BUCompact_p(numbers_io, totalPoints, predicate, address, temp);//150-300micro

			//Swap result with input.
			swap = temp;
			temp = numbers_io;
			numbers_io = swap;

			//Closely spaced zpoints tend to use most significant bits, so we can't break out early.
		}
		//Total loop time around 7 to 10 milli...

		stopBenchmark();
		return error;
	}
#endif

	// Radix Sort Pairs by Key
#ifndef OpenCL
	// NOTE: does not take into account negative numbers
	inline cl_int RadixSortPairsByKey(
		cl::Buffer &keys_io,
		cl::Buffer &values_io,
		cl_int size)
	{
		startBenchmark();
		cl_int error = 0;
		const cl_int globalSize = nextPow2(size);

		cl::Buffer predicate, address, tempKeys, tempValues, swap;
		error |= CLFW::get(address, "radixAddress", sizeof(cl_int)*(globalSize));
		error |= CLFW::get(tempKeys, "tempRadixKeys", sizeof(cl_int)*globalSize);
		error |= CLFW::get(tempValues, "tempRadixValues", sizeof(cl_int)*globalSize);

		if (error != CL_SUCCESS)
			return error;

		//For each bit
		for (cl_int index = 0; index < sizeof(cl_int) * 8; index++) {
			//Predicate the 0's and 1's
			error |= PredicateByBit_p(keys_io, index, 0, size, "rdxsrtbyky", predicate);

			//Scan the predication buffers.
			error |= StreamScan_p(predicate, globalSize, "RSKBVI", address);

			//Compacting
			error |= Compact_p(keys_io, predicate, address, size, tempKeys);
			error |= Compact_p(values_io, predicate, address, size, tempValues);

			//Swap result with input.
			swap = tempKeys;
			tempKeys = keys_io;
			keys_io = swap;

			swap = tempValues;
			tempValues = values_io;
			values_io = swap;
			cl_uint gpuSum;
			//TODO: replace break out code. This was causing differences in running
			//on John's mac between CPU and GPU.
			//if (index % 8 == 0) {
			//    //Checking order is expensive, but can save lots if we can break early.
			//    CheckOrder(unsortedKeys_i, gpuSum, size);
			//    if (gpuSum == 0) break;
			//}
		}

		stopBenchmark();
		return error;
	}
#endif
#pragma endregion

	// Radix Sort BU/Int pairs by Key
#ifndef OpenCL
	// NOTE: does not take into account negative numbers
	inline cl_int RadixSortBUIntPairsByKey(
		cl::Buffer &keys_io,
		cl::Buffer &values_io,
		cl_int mbits,
		cl_int size)
	{
		startBenchmark();
		cl_int error = 0;
		const cl_int globalSize = nextPow2(size);

		cl::Buffer predicate, address, tempKeys, tempValues, swap;
		error |= CLFW::get(address, "radixAddress", sizeof(cl_int)*(globalSize));
		error |= CLFW::get(tempKeys, "tempRadixKeys", sizeof(BigUnsigned)*globalSize);
		error |= CLFW::get(tempValues, "tempRadixValues", sizeof(cl_int)*globalSize);

		if (error != CL_SUCCESS)
			return error;

		//For each bit
		for (cl_int index = 0; index < mbits; index++) {
			//Predicate the 0's and 1's
			error |= PredicateBUByBit_p(keys_io, index, 0, size, "rdxsrtbyky", predicate);

			//Scan the predication buffers.
			error |= StreamScan_p(predicate, globalSize, "RSKBVI", address);

			//Compacting
			error |= BUCompact_p(keys_io, size, predicate, address, tempKeys);
			error |= Compact_p(values_io, predicate, address, size, tempValues);

			//Swap result with input.
			swap = tempKeys;
			tempKeys = keys_io;
			keys_io = swap;

			swap = tempValues;
			tempValues = values_io;
			values_io = swap;
			cl_uint gpuSum;
			//TODO: replace break out code. This was causing differences in running
			//on John's mac between CPU and GPU.
			//if (index % 8 == 0) {
			//    //Checking order is expensive, but can save lots if we can break early.
			//    CheckOrder(unsortedKeys_i, gpuSum, size);
			//    if (gpuSum == 0) break;
			//}
		}

		stopBenchmark();
		return error;
	}
#endif
#pragma endregion

	/* Z-Order Kernels*/
#pragma region Z-Order Kernels

	// Quantize Points
#ifdef OpenCL
	__kernel void QuantizePointsKernel(
		__global floatn *points,
		__global intn *quantizePoints,
		const floatn bbMinimum,
		const float bbMaxWidth,
		const int reslnWidth
	)
	{
		const size_t gid = get_global_id(0);
		const floatn point = points[gid];
		quantizePoints[gid] = QuantizePoint(&point, &bbMinimum, reslnWidth, bbMaxWidth);
	}
#else
	inline cl_int QuantizePoints_p(
		cl::Buffer &points_i,
		cl_uint numPoints,
		const BoundingBox bb,
		const int reslnWidth,
		string uniqueString,
		cl::Buffer &qPoints_o)
	{
		startBenchmark();
		cl_int error = 0;
		cl_int roundSize = nextPow2(numPoints);
		error |= CLFW::get(qPoints_o, uniqueString + "qpts", sizeof(intn)*roundSize);
		cl::Kernel kernel = CLFW::Kernels["QuantizePointsKernel"];
		error |= kernel.setArg(0, points_i);
		error |= kernel.setArg(1, qPoints_o);
		error |= kernel.setArg(2, bb.minimum);
		error |= kernel.setArg(3, bb.maxwidth);
		error |= kernel.setArg(4, reslnWidth);
		error |= CLFW::DefaultQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numPoints), cl::NullRange);
		stopBenchmark();
		return error;
	}

	inline cl_int QuantizePoints_s(
		vector<floatn> &points_i,
		const BoundingBox bb,
		const int reslnWidth,
		vector<intn> &qPoints_o
	) {
		qPoints_o.resize(points_i.size());
		for (int i = 0; i < points_i.size(); ++i) {
			qPoints_o[i] = QuantizePoint(&points_i[i], &bb.minimum, reslnWidth, bb.maxwidth);
		}
		return CL_SUCCESS;
	}
#endif

	// Points To Morton
#ifdef OpenCL
	__kernel void PointsToMortonKernel(
		__global BigUnsigned *inputBuffer,
		__global intn *points,
		const cl_int size,
		const cl_int bits
	)
	{
		const size_t gid = get_global_id(0);
		const size_t lid = get_local_id(0);
		BigUnsigned tempBU;
		intn tempPoint = points[gid];

		if (gid < size) {
			xyz2z(&tempBU, tempPoint, bits);
		}
		else {
			initBlkBU(&tempBU, 0);
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
		inputBuffer[gid] = tempBU;
	}
#else
	inline cl_int QPointsToZPoints_p(
		cl::Buffer &qpoints,
		cl_int totalPoints,
		cl_int reslnBits,
		string uniqueString,
		cl::Buffer &zpoints
	)
	{
		startBenchmark();
		cl_int error = 0;
		cl_int globalSize = nextPow2(totalPoints);
		bool old;
		error |= CLFW::get(zpoints, uniqueString + "zpts", globalSize * sizeof(BigUnsigned), old, CLFW::DefaultContext, CL_MEM_READ_ONLY);
		cl::Kernel kernel = CLFW::Kernels["PointsToMortonKernel"];
		error |= kernel.setArg(0, zpoints);
		error |= kernel.setArg(1, qpoints);
		error |= kernel.setArg(2, totalPoints);
		error |= kernel.setArg(3, reslnBits);
		error |= CLFW::DefaultQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
		//vector<BigUnsigned> zpoints_vec;
		//DownloadZPoints(zpoints_vec, zpoints, globalSize);
		//for (int i = 0; i < globalSize; ++i) {
		//  cout << buToString(zpoints_vec[i]) << endl;
		//}
		stopBenchmark();
		return error;
	};

	inline cl_int QPointsToZPoints_s(vector<intn> &input, cl_int reslnBits, vector<BigUnsigned> &result) {
		startBenchmark();
		for (int gid = 0; gid < input.size(); ++gid) {
			if (gid < input.size()) {
				xyz2z(&result[gid], input[gid], reslnBits);
			}
			else {
				initBlkBU(&result[gid], 0);
			}
		}
		stopBenchmark();
		return 0;
	}
#endif
#pragma endregion

	/* Hybrid Kernels */
#pragma region Hybrid Kernels

	// Unique Sorted
#ifndef OpenCL
	inline cl_int UniqueSorted(
		cl::Buffer &input_io,
		cl_int originalSize,
		string uniqueString,
		cl_int &newSize
	) {
		startBenchmark();
		int globalSize = nextPow2(originalSize);
		cl_int error = 0;

		cl::Buffer predicate, address, intermediate, result;
		error = CLFW::get(predicate, uniqueString + "uniqpred", sizeof(cl_int)*(globalSize));
		error |= CLFW::get(address, uniqueString + "uniqaddr", sizeof(cl_int)*(globalSize));
		error |= CLFW::get(result, uniqueString + "uniqresult", sizeof(BigUnsigned) * globalSize);

		error |= PredicateUnique_p(input_io, predicate, originalSize);
		error |= StreamScan_p(predicate, originalSize, uniqueString + "uniqI", address);
		error |= BUSingleCompact(input_io, result, predicate, address, originalSize);
		input_io = result;

		error |= CLFW::DefaultQueue.enqueueReadBuffer(address, CL_TRUE, (sizeof(cl_int)*originalSize - (sizeof(cl_int))), sizeof(cl_int), &newSize);
		stopBenchmark();
		return error;
	}

	inline cl_int UniqueSortedBUIntPair(
		cl::Buffer &keys_io,
		cl::Buffer &values_io,
		cl_int originalSize,
		string uniqueString,
		cl_int &newSize
		) {
		startBenchmark();
		int globalSize = nextPow2(originalSize);
		cl_int error = 0;

		cl::Buffer predicate, address, intermediate, result_keys, result_vals;
		error = CLFW::get(predicate, uniqueString + "uniqpred", sizeof(cl_int)*(globalSize));
		error |= CLFW::get(address, uniqueString + "uniqaddr", sizeof(cl_int)*(globalSize));
		error |= CLFW::get(result_keys, uniqueString + "uniqkresult", sizeof(BigUnsigned) * globalSize);
		error |= CLFW::get(result_vals, uniqueString + "uniqvresult", sizeof(cl_int) * globalSize);

		error |= PredicateUnique_p(keys_io, predicate, originalSize);
		error |= StreamScan_p(predicate, originalSize, uniqueString + "uniqI", address);
		error |= BUCompact_p(keys_io, originalSize, predicate, address, result_keys);
		error |= Compact_p(values_io, predicate, address, originalSize, result_vals);
		keys_io = result_keys;
		values_io = result_vals;

		error |= CLFW::DefaultQueue.enqueueReadBuffer(address, CL_TRUE, (sizeof(cl_int)*originalSize - (sizeof(cl_int))), sizeof(cl_int), &newSize);
		stopBenchmark();
		return error;
	}
#endif

	// Check Order
#ifdef OpenCL
	__kernel
		void CheckOrder(
			__global int *numbers,
			__local cl_int* scratch,
			__const int numZPoints,
			__const int nextPowerOfTwo,
			__global cl_int* result)
	{
		//Init
		int global_index = get_global_id(0);
		int local_index = get_local_id(0);
		cl_int accumulator = 0;

		// Loop sequentially over chunks of the BU vector, comparing along the way
		while (global_index < nextPowerOfTwo) {
			if (global_index < numZPoints - 1)
			{
				int mine = numbers[global_index];
				int other = numbers[global_index + 1];
				accumulator += max(mine > other, 0);
			}
			global_index += get_global_size(0);
			//Enforce coherency
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		// Perform parallel reduction
		scratch[local_index] = accumulator;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
			if (local_index < offset)
				scratch[local_index] = scratch[local_index] + scratch[local_index + offset];
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if (local_index == 0) {
			result[get_group_id(0)] = scratch[0];
		}
	}
#else
	inline cl_int CheckOrder(cl::Buffer &numbers, cl_uint& gpuSum, cl_int size) {
		startBenchmark();
		if (size < 0) return CL_INVALID_ARG_SIZE;
		else if (size < 2) {
			gpuSum = 0;
			return CL_SUCCESS;
		}
		cl_int nextPowerOfTwo = pow(2, ceil(log(size) / log(2)));
		cl::Kernel &kernel = CLFW::Kernels["CheckOrder"];
		cl::CommandQueue &queue = CLFW::DefaultQueue;

		//Each thread processes 2 items in the reduce.
		int globalSize = nextPowerOfTwo / 2;

		//Brent's theorem
		/* int itemsPerThread = pow(2, ceil(log(log(globalSize)) / log(2)));
		if ((itemsPerThread < globalSize) && itemsPerThread != 0) {
		globalSize /= itemsPerThread;
		}*/

		int suggestedLocal = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLFW::DefaultDevice);
		int localSize = std::min(globalSize, suggestedLocal);

		cl::Buffer reduceResult;
		cl_int resultSize = nextPow2(nextPowerOfTwo / localSize);
		cl_int error = CLFW::get(reduceResult, "reduceResult", resultSize * sizeof(cl_uint));

		error |= kernel.setArg(0, numbers);
		error |= kernel.setArg(1, cl::__local(localSize * sizeof(cl_uint)));
		error |= kernel.setArg(2, size);
		error |= kernel.setArg(3, nextPowerOfTwo);
		error |= kernel.setArg(4, reduceResult);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize));

		//If multiple workgroups ran, we need to do a second level reduction.
		if (suggestedLocal <= globalSize) {
			cl::Kernel &kernel = CLFW::Kernels["reduce"];
			error |= kernel.setArg(0, reduceResult);
			error |= kernel.setArg(1, cl::__local(localSize * sizeof(cl_uint)));
			error |= kernel.setArg(2, resultSize);
			error |= kernel.setArg(3, reduceResult);
			error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(localSize / 2), cl::NDRange(localSize / 2));
		}
		error |= queue.enqueueReadBuffer(reduceResult, CL_TRUE, 0, sizeof(cl_uint), &gpuSum);
		stopBenchmark();
		return error;
	}
#endif
#pragma endregion

	/* Tree Building Kernels */
#pragma region Tree Building Kernels

	// Compute Local Splits
#ifdef OpenCL
	__kernel void ComputeLocalSplitsKernel(
		__global cl_int* local_splits,
		__global BrtNode* I,
		cl_int colored,
		__global cl_int* colors,
		const int size
	)
	{
		const size_t gid = get_global_id(0);
		if (size > 0 && gid == 0) {
			local_splits[0] = 1 + I[0].lcp.len/ DIM;
		}
		if (gid < size - 1) {
			ComputeLocalSplits(local_splits, I, colored, colors, gid);
		}
	}
#else
	inline cl_int ComputeLocalSplits_p(
		cl::Buffer &internalBRTNodes_i, 
		cl_int totalBRT, 
		cl_int colored, 
		cl::Buffer &colors,
		string uniqueString, 
		cl::Buffer &localSplits_o) {
		startBenchmark();
		cl_int globalSize = nextPow2(totalBRT);
		cl::Kernel &kernel = CLFW::Kernels["ComputeLocalSplitsKernel"];
		cl::CommandQueue &queue = CLFW::DefaultQueue;

		bool isOld;
		cl::Buffer zeroBuffer;

		cl_int error = CLFW::get(localSplits_o, uniqueString + "localSplits", sizeof(cl_int) * globalSize);
		error |= CLFW::get(zeroBuffer, uniqueString + "zeroBuffer", sizeof(cl_int) * globalSize, isOld);

		//Fill any new zero buffers with zero. Then initialize localSplits with zero.
		if (!isOld) {
			cl_int zero = 0;
			error |= queue.enqueueFillBuffer<cl_int>(zeroBuffer, { zero }, 0, sizeof(cl_int) * globalSize);
		}
		error |= queue.enqueueCopyBuffer(zeroBuffer, localSplits_o, 0, 0, sizeof(cl_int) * globalSize);

		error |= kernel.setArg(0, localSplits_o);
		error |= kernel.setArg(1, internalBRTNodes_i);
		error |= kernel.setArg(2, colored);
		error |= kernel.setArg(3, colors);
		error |= kernel.setArg(4, totalBRT);

		error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
		stopBenchmark();
		return error;
	}

	inline cl_int ComputeLocalSplits_s(
		vector<BrtNode> &I, 
		bool colored, 
		vector<cl_int> colors, 
		vector<cl_int> &local_splits, 
		const cl_int size) 
	{
		startBenchmark();
		if (size > 0) {
			local_splits[0] = 1 + I[0].lcp.len / DIM;
		}
		for (int i = 0; i < size - 1; ++i) {
			ComputeLocalSplits(local_splits.data(), I.data(), colored, colors.data(), i);
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	// Build Binary Radix Tree
#ifdef OpenCL
	__kernel void BuildBinaryRadixTreeKernel(
		__global BrtNode *I,
		__global BigUnsigned* mpoints,
		int mbits,
		int size
	)
	{
		BuildBinaryRadixTree(I, nullptr, mpoints, nullptr, mbits, size, false, get_global_id(0));
	}
#else
	inline cl_int BuildBinaryRadixTree_p(
		cl::Buffer &zpoints_i,
		cl_int totalUniquePoints,
		cl_int mbits,
		string uniqueString,
		cl::Buffer &internalBRTNodes_o
	) {
		startBenchmark();
		cl::Kernel &kernel = CLFW::Kernels["BuildBinaryRadixTreeKernel"];
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl_int globalSize = nextPow2(totalUniquePoints);
		bool isOld;
		cl::Buffer zeroBRTNodes;
		cl_int error = CLFW::get(internalBRTNodes_o, uniqueString + "brt", sizeof(BrtNode)* (globalSize));
		error |= CLFW::get(zeroBRTNodes, uniqueString + "brtzero", sizeof(BrtNode)* (globalSize), isOld);
		if (!isOld) {
			BrtNode b = { 0 };
			queue.enqueueFillBuffer<BrtNode>(zeroBRTNodes, { b }, 0, sizeof(BrtNode) * globalSize);
		}
		error |= queue.enqueueCopyBuffer(zeroBRTNodes, internalBRTNodes_o, 0, 0, sizeof(BrtNode)* (globalSize));
		error |= kernel.setArg(0, internalBRTNodes_o);
		error |= kernel.setArg(1, zpoints_i);
		error |= kernel.setArg(2, mbits);
		error |= kernel.setArg(3, totalUniquePoints);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
		stopBenchmark();

		return error;
	}

	inline cl_int BuildBinaryRadixTree_s(vector<BigUnsigned> &zpoints, cl_int mbits, vector<BrtNode> &internalBRTNodes) {
		internalBRTNodes.resize(zpoints.size() - 1);
		startBenchmark();
		for (int i = 0; i < zpoints.size() - 1; ++i) {
			BuildBinaryRadixTree(internalBRTNodes.data(), nullptr, zpoints.data(), nullptr, mbits, zpoints.size(), false, i);
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	//Color Binary Radix Tree
#ifdef OpenCL
	__kernel void BuildColoredBinaryRadixTreeKernel(
		__global BrtNode *I,
		__global cl_int *IColors,
		__global BigUnsigned* mpoints,
		__global cl_int *pointColors,
		int mbits,
		int size
		)
	{
		BuildBinaryRadixTree(I, IColors, mpoints, pointColors, mbits, size, true, get_global_id(0));
	}
#else
	inline cl_int BuildColoredBinaryRadixTree_p(
		cl::Buffer &zpoints_i,
		cl::Buffer  &pointColors_i,
		cl_int totalUniquePoints,
		cl_int mbits,
		string uniqueString,
		cl::Buffer  &brt_o,
		cl::Buffer  &brtColors_o)
	{
		startBenchmark();
		cl::Kernel &kernel = CLFW::Kernels["BuildColoredBinaryRadixTreeKernel"];
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl_int globalSize = nextPow2(totalUniquePoints);
		cl::Buffer zeroBrtNodes;
		bool isOld;

		cl_int error = CLFW::get(brt_o, uniqueString + "brt", sizeof(BrtNode)* (globalSize));
		error |= CLFW::get(zeroBrtNodes, uniqueString + "brtzero", sizeof(BrtNode)* (globalSize), isOld);
		if (!isOld) {
			BrtNode b = { 0 };
			queue.enqueueFillBuffer<BrtNode>(zeroBrtNodes, { b }, 0, sizeof(BrtNode) * globalSize);
		}
		error |= queue.enqueueCopyBuffer(zeroBrtNodes, brt_o, 0, 0, sizeof(BrtNode)* (globalSize));
		error |= CLFW::get(brtColors_o, uniqueString + "brtc", sizeof(cl_int)* (globalSize));
		error |= kernel.setArg(0, brt_o);
		error |= kernel.setArg(1, brtColors_o);
		error |= kernel.setArg(2, zpoints_i);
		error |= kernel.setArg(3, pointColors_i);
		error |= kernel.setArg(4, mbits);
		error |= kernel.setArg(5, totalUniquePoints);

		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(totalUniquePoints - 1), cl::NullRange);
		stopBenchmark();
		return error;
	}

	inline cl_int BuildColoredBinaryRadixTree_s(
		vector<BigUnsigned> &zpoints_i, 
		vector<cl_int> &pointColors_i, 
		cl_int mbits, 
		vector<BrtNode> &brt_o, 
		vector<cl_int> &brtColors_o) 
	{
		brt_o.resize(zpoints_i.size() - 1);
		brtColors_o.resize(zpoints_i.size() - 1);
		startBenchmark();
		for (int i = 0; i < zpoints_i.size() - 1; ++i) {
			BuildBinaryRadixTree(
				brt_o.data(), 
				brtColors_o.data(), 
				zpoints_i.data(), 
				pointColors_i.data(), 
				mbits, 
				zpoints_i.size(), 
				true, 
				i);
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	//Propagate Binary Radix Tree Colors
#ifdef OpenCL
	__kernel void PropagateBRTColorsKernel(
		__global BrtNode *brt_i,
		volatile __global cl_int *brtColors_io,
		cl_int totalBrtNodes
		)
	{
		cl_int gid = get_global_id(0);
		//for (int gid = 0; gid < totalBrtNodes; gid++) {

			cl_int index = gid;
			BrtNode node = brt_i[gid];

			//Only run BRT nodes with leaves
			if (node.left_leaf || node.right_leaf) {
				cl_int currentColor = brtColors_io[gid];

				//Traverse up the tree
				while (index != 0) {
					index = node.parent;
					node = brt_i[index];

					//If the parent has no color, paint it and exit.
					cl_int r = atomic_cmpxchg(&brtColors_io[index], -1, currentColor);
					if (r == -1)  break;
					// else if our colors don't match, mark it
					else if (r != currentColor) {
						if (r != -2) brtColors_io[index] = -2;
						currentColor = -2;
					}
				}
			}
		//}
	}
#else
	inline cl_int PropagateBRTColors_p(
		cl::Buffer &brt_i,
		cl::Buffer &brtColors_io,
		cl_int totalElements,
		string uniqueString)
	{
		startBenchmark();
		cl::Kernel &kernel = CLFW::Kernels["PropagateBRTColorsKernel"];
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl_int globalSize = nextPow2(totalElements);
		cl_int error = 0;
		error |= kernel.setArg(0, brt_i);
		error |= kernel.setArg(1, brtColors_io);
		error |= kernel.setArg(2, totalElements);

		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(totalElements), cl::NullRange);
		stopBenchmark();
		return error;
	}

	inline cl_int PropagateBRTColors_s(
		vector<BrtNode> &brt_i,
		vector<cl_int> &brtColors_io)
	{
		startBenchmark();
		for (int gid = 0; gid < brt_i.size(); ++gid) {

			cl_int index = gid;
			BrtNode node = brt_i[gid];

			//Only run BRT nodes with leaves
			if (node.left_leaf || node.right_leaf) {
				cl_int currentColor = brtColors_io[gid];

				//Traverse up the tree
				while (index != 0) {
					index = node.parent;
					node = brt_i[index];

					//If the parent has no color, paint it and exit.
						//atomic_cmpxchg(&brtColors_io[index], -1, currentColor);
					cl_int r = brtColors_io[index];
					if (brtColors_io[index] == -1) brtColors_io[index] = currentColor;

					if (r == -1)  break;
					// else if our colors don't match, mark it
					else if (r != currentColor) {
						if (r != -2) brtColors_io[index] = -2;
						currentColor = -2;
					}
				}
			}
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	// Init Octree
#ifdef OpenCL
	__kernel void BRT2OctreeKernel_init(
		__global OctNode *octree
	) {
		brt2octree_init(octree, get_global_id(0));
	}
#else
	inline cl_int InitOctree(cl::Buffer &octree_i, cl_int octreeSize) {
		startBenchmark();
		cl::Kernel &kernel = CLFW::Kernels["BRT2OctreeKernel_init"];
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl_int error = 0;
		error |= kernel.setArg(0, octree_i);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(octreeSize), cl::NullRange);
		stopBenchmark();
		return error;
	}
#endif

	// Binary Radix To Octree
#ifdef OpenCL
	__kernel void BRT2OctreeKernel(
		__global BrtNode *I,
		const cl_int totalBrtNodes,
		__global OctNode *octree,
		const cl_int totalOctNodes,
		__global cl_int *localSplits,
		__global cl_int *prefixSums,
		__global cl_int *flags
	) {
		const int gid = get_global_id(0);
		brt2octree(I, totalBrtNodes, octree, totalOctNodes, localSplits, prefixSums, flags, gid);
	}
#else
	inline cl_int BinaryRadixToOctree_p(
		cl::Buffer &brt_i,
		bool colored,
		cl::Buffer colors_i,
		cl_int totalBRTNode,
		string uniqueString,
		cl::Buffer &octree_o,
		int &octreeSize_o
	) {
		cl_int error = 0;

		if (totalBRTNode < 1) return CL_SUCCESS;
		else if (totalBRTNode == 1) {
			error |= CLFW::get(octree_o, uniqueString + "octree", sizeof(OctNode));
			OctNode root = {-1, -1, -1, -1, -1, -1 -1, -1, -1 };
			error |= CLFW::Upload<OctNode>(root, 0, octree_o); 
			octreeSize_o = 1;
			return error;
		}
		startBenchmark();
		int globalSize = nextPow2(totalBRTNode);
		cl::Kernel &kernel = CLFW::Kernels["BRT2OctreeKernel"];
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl::Buffer localSplits, scannedSplits, flags;
		bool isOld;
		error |= CLFW::get(scannedSplits, uniqueString + "scannedSplits", sizeof(cl_int) * globalSize);
		error |= CLFW::get(flags, uniqueString + "flags", nextPow2(totalBRTNode) * sizeof(cl_int), isOld);
		if (isOld) error |= CLFW::DefaultQueue.enqueueFillBuffer<cl_int>(flags, { 0 }, 0, sizeof(cl_int) * nextPow2(totalBRTNode));
		
		error |= ComputeLocalSplits_p(brt_i, totalBRTNode, colored, colors_i, uniqueString, localSplits);
		
		error |= StreamScan_p(localSplits, globalSize, uniqueString + "octreeI", scannedSplits);
		//Read in the required octree size
		cl_int octreeSize;
		error |= CLFW::DefaultQueue.enqueueReadBuffer(scannedSplits, CL_TRUE,
			sizeof(cl_int)*(totalBRTNode - 1), sizeof(cl_int), &octreeSize);
		cl_int roundOctreeSize = nextPow2(octreeSize);

		//Create an octree buffer.
		error |= CLFW::get(octree_o, uniqueString + "octree", sizeof(OctNode) * roundOctreeSize);

		//use the scanned splits & brt to create octree.
		error |= InitOctree(octree_o, octreeSize);
		error |= kernel.setArg(0, brt_i);
		error |= kernel.setArg(1, totalBRTNode);
		error |= kernel.setArg(2, octree_o);
		error |= kernel.setArg(3, totalBRTNode);
		error |= kernel.setArg(4, localSplits);
		error |= kernel.setArg(5, scannedSplits);
		error |= kernel.setArg(6, flags);

		/* Skip the root */
		error |= queue.enqueueNDRangeKernel(kernel, cl::NDRange(1), cl::NDRange(totalBRTNode - 1), cl::NullRange);
		octreeSize_o = octreeSize;
		stopBenchmark();
		return error;
	}
#endif
#ifndef OpenCL
	inline cl_int BinaryRadixToOctree_s(
		vector<BrtNode> &internalBRTNodes_i, 
		bool colored,
		vector<cl_int> brtColors_i,
		vector<OctNode> &octree_o
		) {
		startBenchmark();
		int size = internalBRTNodes_i.size();
		vector<cl_int> localSplits(size);
		ComputeLocalSplits_s(internalBRTNodes_i, colored, brtColors_i, localSplits, size);

		vector<cl_int> prefixSums(size);
		StreamScan_s(localSplits, prefixSums);

		vector<cl_int> flags(internalBRTNodes_i.size(), 0);

		const int octreeSize = prefixSums[size - 1];
		octree_o.resize(octreeSize);
		octree_o[0].parent = -1;
		octree_o[0].level = 0;
		for (int i = 0; i < octreeSize; ++i)
			brt2octree_init(octree_o.data(), i);
		for (int brt_i = 1; brt_i < size - 1; ++brt_i)
			brt2octree(internalBRTNodes_i.data(), internalBRTNodes_i.size(), 
				octree_o.data(), octree_o.size(), localSplits.data(), 
				prefixSums.data(), flags.data(), brt_i);
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	//  // Build Octree
	//#ifndef OpenCL
	//  inline cl_int BuildOctree_s(const vector<intn>& points, vector<OctNode> &octree, int bits, int mbits) {
	//    startBenchmark();
	//    if (points.empty()) {
	//      throw logic_error("Zero points not supported");
	//      return -1;
	//    }
	//    int numPoints = points.size();
	//    int roundNumPoints = nextPow2(points.size());
	//    vector<BigUnsigned> zpoints(roundNumPoints);
	//
	//    //Points to Z Order
	//    PointsToMorton_s(points.size(), bits, (intn*)points.data(), zpoints.data());
	//
	//    //Sort and unique Z points
	//    sort(zpoints.rbegin(), zpoints.rend(), weakCompareBU);
	//    numPoints = unique(zpoints.begin(), zpoints.end(), weakEqualsBU) - zpoints.begin();
	//
	//    //Build BRT
	//    vector<BrtNode> I(numPoints - 1);
	//    BuildBinaryRadixTree_s(zpoints.data(), I.data(), numPoints, mbits);
	//
	//    //Build Octree
	//    BinaryRadixToOctree_s(I, octree, numPoints);
	//    stopBenchmark();
	//    return CL_SUCCESS;
	//  }
	//#endif
	//#ifndef OpenCL
	//  inline cl_int BuildOctree_p(cl::Buffer zpoints_i, cl_int numZPoints, cl::Buffer &octree_o, string octreeName, int &octreeSize_o, int bits, int mbits) {
	//    startBenchmark();
	//    int currentSize = numZPoints;
	//    cl_int error = 0;
	//    cl::Buffer sortedZPoints, internalBRTNodes;
	//    error |= RadixSortBigUnsigned_p(zpoints_i, sortedZPoints, currentSize, mbits);
	//    assert(error == CL_SUCCESS);
	//    error |= UniqueSorted(sortedZPoints, currentSize);
	//    assert(error == CL_SUCCESS);
	//    error |= BuildBinaryRadixTree_p(sortedZPoints, internalBRTNodes, currentSize, mbits);
	//    assert(error == CL_SUCCESS);
	//    error |= BinaryRadixToOctree_p(internalBRTNodes, currentSize, octree_o, octreeName, octreeSize_o); //occasionally currentSize is 0...
	//    assert(error == CL_SUCCESS);
	//    stopBenchmark();
	//    return error;
	//  }
	//#endif

		// Compute Leaves
#ifdef OpenCL
	__kernel void ComputeLeavesKernel(
		__global OctNode *octree,
		__global Leaf *leaves,
		__global cl_int *leafPredicates,
		int octreeSize
	)
	{
		const int gid = get_global_id(0);
		ComputeLeaves(octree, leaves, leafPredicates, octreeSize, gid);
	}
#else
	inline cl_int GenerateLeaves_p(
		cl::Buffer &octree_i,
		int octreeSize,
		cl::Buffer &sparseleaves_o,
		cl::Buffer &leafPredicates_o)
	{
		cl_int error = 0;
		error |= CLFW::get(sparseleaves_o, "sleaves", nextPow2(4 * octreeSize) * sizeof(Leaf));
		error |= CLFW::get(leafPredicates_o, "lfPrdcts", nextPow2(4 * octreeSize) * sizeof(cl_int));
		cl::Kernel &kernel = CLFW::Kernels["ComputeLeavesKernel"];

		error |= kernel.setArg(0, octree_i);
		error |= kernel.setArg(1, sparseleaves_o);
		error |= kernel.setArg(2, leafPredicates_o);
		error |= kernel.setArg(3, octreeSize);

		error |= CLFW::DefaultQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(nextPow2(4 * octreeSize)), cl::NullRange);
		return error;
	}
	inline cl_int GenerateLeaves_s(
		vector<OctNode> &octree,
		cl_int octreeSize,
		vector<Leaf> &leaves,
		vector<cl_int> &leafPredicates
	)
	{
		for (int i = 0; i < 4 * octreeSize; ++i )
			ComputeLeaves(octree.data(), leaves.data(), leafPredicates.data(), octreeSize, i);
		return CL_SUCCESS;
	}
#endif

	// Get Leaves
#ifndef OpenCL
	inline cl_int GetLeaves_p(cl::Buffer &octree_i,
		int octreeSize,
		cl::Buffer &Leaves_o,
		int &totalLeaves)
	{
		cl::Buffer sparseLeafParents, leafPredicates, leafAddresses;

		cl_int error = GenerateLeaves_p(octree_i, octreeSize, sparseLeafParents, leafPredicates);
		CLFW::get(leafAddresses, "lfaddrs", nextPow2(octreeSize * 4) * sizeof(cl_int));
		CLFW::get(Leaves_o, "leaves", nextPow2(octreeSize * 4) * sizeof(Leaf));
		error |= Kernels::StreamScan_p(leafPredicates, nextPow2(octreeSize * 4), "lfintrmdt", leafAddresses);
		error |= CLFW::DefaultQueue.enqueueReadBuffer(leafAddresses, CL_TRUE, (sizeof(cl_int)*(octreeSize * 4) - (sizeof(cl_int))), sizeof(cl_int), &totalLeaves);
		error |= Kernels::LeafDoubleCompact(sparseLeafParents, Leaves_o, leafPredicates, leafAddresses, octreeSize * 4);
		return error;
	}
#endif

	// Predicate Duplicate Nodes
#ifdef OpenCL
	__kernel void PredicateDuplicateNodesKernelPart1(
		__global OctNode *origOT,
		__global OctNode *newOT,
		__global cl_int *predicates,
		int newOTSize
	)
	{
		const int gid = get_global_id(0);
		if (gid < newOTSize) {
			PredicateDuplicateNodes(origOT, newOT, predicates, newOTSize, gid);
		}
	}

	__kernel void PredicateDuplicateNodesKernelPart2(
		__global OctNode *origOT,
		__global OctNode *newOT,
		__global cl_int *predicates,
		int newOTSize
	)
	{
		//If the current node is a duplicate...
		const int gid = get_global_id(0);
		if (predicates[gid] != -1) {
			//Update my children's "parent" index to the original node
			OctNode current = origOT[gid];
			for (int i = 0; i < DIM; ++i) {
				if ((current.leaf & (1 << i)) == 0)
				{
					//When I perish, have the original node adopt my children. ;(
					newOT[current.children[i]].parent = predicates[gid];
				}
			}
			predicates[gid] = 0; // Predicate me for deletion
		}
		else {
			predicates[gid] = 1; // Compact me to the left.
		}
	}
#else
	inline cl_int PredicateDuplicateNodes_p(cl::Buffer origOT_i, cl::Buffer newOT_i, int newOTSize, cl::Buffer &duplicate_o) {
		cl_int error = 0;
		cl::Kernel &kernel = CLFW::Kernels["PredicateDuplicateNodesKernelPart1"];
		CLFW::get(duplicate_o, "dup", sizeof(cl_int) * nextPow2(newOTSize));
		error |= kernel.setArg(0, origOT_i);
		error |= kernel.setArg(1, newOT_i);
		error |= kernel.setArg(2, duplicate_o);
		error |= kernel.setArg(3, newOTSize);
		error |= CLFW::DefaultQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(nextPow2(newOTSize)), cl::NullRange);

		//Nvidia's drivers crash when I run this code in one kernel. 
		cl::Kernel &kernel2 = CLFW::Kernels["PredicateDuplicateNodesKernelPart2"];
		cl_int memsize = kernel2.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(CLFW::DefaultDevice);
		error |= kernel2.setArg(0, origOT_i);
		error |= kernel2.setArg(1, newOT_i);
		error |= kernel2.setArg(2, duplicate_o);
		error |= kernel2.setArg(3, newOTSize);
		error |= CLFW::DefaultQueue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(nextPow2(newOTSize)), cl::NullRange);
		return error;
	}
#endif
#pragma endregion

	/* Ambiguous cell resolution kernels */
#pragma region Ambiguous Cell Resolution Kernels

	//Get Line LCPs
#ifdef OpenCL
	__kernel void GetLineLCPKernel(
		__global Line* lines,
		__global BigUnsigned* zpoints,
		__global LCP* LineLCPs,
		const int mbits
	) {
		const int gid = get_global_id(0);
		GetLCPFromLine(lines, zpoints, LineLCPs, mbits, gid);
	}
#else
	inline cl_int GetLineLCPs_p(
		cl::Buffer &linesBuffer_i,
		cl_int totalLines,
		cl::Buffer &zpoints_i,
		int bitsPerZPoint,
		cl::Buffer &LineLCPs_o
	)
	{
		startBenchmark();
		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["GetLineLCPKernel"];
		cl_int error = 0;

		cl_int roundSize = nextPow2(totalLines);
		error |= CLFW::get(LineLCPs_o, "LineLCPs", roundSize * sizeof(LCP));
		error |= kernel->setArg(0, linesBuffer_i);
		error |= kernel->setArg(1, zpoints_i);
		error |= kernel->setArg(2, LineLCPs_o);
		error |= kernel->setArg(3, bitsPerZPoint);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(totalLines), cl::NullRange);

		stopBenchmark();
		return error;
	}
	inline cl_int GetLineLCP_s(
		vector<Line> &lines,
		vector<BigUnsigned> &zpoints,
		int mbits,
		vector<LCP> &LineLCPs
	) {
		startBenchmark();
		for (int i = 0; i < lines.size(); ++i) {
			GetLCPFromLine(lines.data(), zpoints.data(), LineLCPs.data(), mbits, i);
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

#ifdef OpenCL
	__kernel void InitializeFacetIndicesKernel(
		__global int* facetIndices
	) {
		facetIndices[get_global_id(0)] = get_global_id(0);
	}
#else
	inline cl_int InitializeFacetIndices_p(
		cl_int totalFacets,
		cl::Buffer &facetIndices_o
	)
	{
		startBenchmark();
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl::Kernel &kernel = CLFW::Kernels["InitializeFacetIndicesKernel"];
		cl_int error = 0;

		cl_int roundSize = nextPow2(totalFacets);
		error |= CLFW::get(facetIndices_o, "facetIndices", roundSize * sizeof(cl_int));
		error |= kernel.setArg(0, facetIndices_o);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(totalFacets), cl::NullRange);

		stopBenchmark();
		return error;
	}
#endif

	// Look Up Octnode From Line LCP
#ifndef OpenCL
	inline cl_int LookUpOctnodeFromLCP_s(LCP* LCPs, OctNode *octree, int* FacetToOctree, cl_int numLCPs) {
		startBenchmark();
		for (int i = 0; i < numLCPs; ++i) {
			FacetToOctree[i] = getOctNode(LCPs[i].bu, LCPs[i].len, octree, &octree[0]);
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif
#ifdef OpenCL
	__kernel void LookUpOctnodeFromLCPKernel(
		__global LCP* LCPs,
		__global OctNode* octree,
		__global int* FacetToOctree
	) {
		const int gid = get_global_id(0);
		LCP LCP = LCPs[gid];
		__local OctNode root;
		if (get_local_id(0) == 0) {
			root = octree[0];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		int index = getOctNode(LCP.bu, LCP.len, octree, &root);
    
		FacetToOctree[gid] = index;
	}
#else
	inline cl_int LookUpOctnodeFromLCP_p(
		cl::Buffer &LCPs_i,
		cl_int numLCPs,
		cl::Buffer &octree_i,
		cl::Buffer &LCPToOctnode_o
	) {
		startBenchmark();

		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["LookUpOctnodeFromLCPKernel"];
		int roundNumber = nextPow2(numLCPs);
		cl_int error = CLFW::get(
			LCPToOctnode_o, "LCPToOctnode", sizeof(cl_int)* (roundNumber));

		error |= kernel->setArg(0, LCPs_i);
		error |= kernel->setArg(1, octree_i);
		error |= kernel->setArg(2, LCPToOctnode_o);
		error |= queue->enqueueNDRangeKernel( //Around 160 microseconds on gtx 1070
			*kernel, cl::NullRange, cl::NDRange(numLCPs), cl::NullRange);

		stopBenchmark();
		return error;
	}

	inline cl_int LookUpOctnodeFromLCP_s(
		vector<LCP> &LCPs_i,
		vector<OctNode> &octree_i,
		vector<cl_int> &LCPToOctnode_o
	) {
		LCPToOctnode_o.resize(LCPs_i.size());
		OctNode root = octree_i[0];
		for ( int gid = 0; gid < LCPs_i.size(); ++gid) {
			LCP LCP = LCPs_i[gid];
			int index = getOctNode(LCP.bu, LCP.len, octree_i.data(), &root);
			LCPToOctnode_o[gid] = index;
		}
		return CL_SUCCESS;
	}
#endif

	// Get Facet Pairs
#ifndef OpenCL
	inline cl_int GetFacetPairs_s(cl_int* FacetToOctree, Pair *facetPairs, cl_int numLines) {
		startBenchmark();
		for (int i = 0; i < numLines; ++i) {
			int leftNeighbor = (i == 0) ? -1 : FacetToOctree[i - 1];
			int rightNeighbor = (i == numLines - 1) ? -1 : FacetToOctree[i + 1];
			int me = FacetToOctree[i];
			//If my left neighbor doesn't go to the same octnode I go to
			if (leftNeighbor != me) {
				//Then I am the first LCP/Facet belonging to my octnode
				facetPairs[me].first = i;
			}
			//If my right neighbor doesn't go the the same octnode I go to
			if (rightNeighbor != me) {
				//Then I am the last LCP/Facet belonging to my octnode
				facetPairs[me].last = i;
			}
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif
#ifdef OpenCL
	__kernel void GetFacetPairsKernel(
		__global int* FacetToOctree,
		__global Pair *facetPairs,
		int numLines
	)
	{
		const int gid = get_global_id(0);
		int leftNeighbor = (gid == 0) ? -1 : FacetToOctree[gid - 1];
		int rightNeighbor = (gid == numLines - 1) ? -1 : FacetToOctree[gid + 1];
		int me = FacetToOctree[gid];
		//If my left neighbor doesn't go to the same octnode I go to
		if (leftNeighbor != me) {
			//Then I am the first LCP/Facet belonging to my octnode
			facetPairs[me].first = gid;
		}
		//If my right neighbor doesn't go the the same octnode I go to
		if (rightNeighbor != me) {
			//Then I am the last LCP/Facet belonging to my octnode
			facetPairs[me].last = gid;
		}
	}
#else
	inline cl_int GetLCPBounds_p(
		cl::Buffer &orderedNodeIndices_i,
		cl_int numLines,
		cl_int octreeSize,
		cl::Buffer &facetPairs_o
	) {
		startBenchmark();

		cl::CommandQueue *queue = &CLFW::DefaultQueue;
		cl::Kernel *kernel = &CLFW::Kernels["GetFacetPairsKernel"];
		int roundNumber = nextPow2(octreeSize);
		cl_int error = CLFW::get(facetPairs_o, "facetPairs", sizeof(Pair)* (roundNumber));
		Pair initialPair = { -1, -1 };
		error |= queue->enqueueFillBuffer<Pair>(facetPairs_o, { initialPair }, 0, sizeof(Pair) * roundNumber);
		error |= kernel->setArg(0, orderedNodeIndices_i);
		error |= kernel->setArg(1, facetPairs_o);
		error |= kernel->setArg(2, numLines);
		error |= queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(numLines), cl::NullRange);

		stopBenchmark();
		return error;
	}
	inline cl_int GetLCPBounds_s(
		vector<cl_int> &facetToOctnode_i,
		cl_int numLines,
		cl_int octreeSize,
		vector<Pair> &facetPairs_o
	) {
		facetPairs_o.resize(octreeSize, {-1,-1});
		for (int gid = 0; gid < numLines; ++gid) {
			int leftNeighbor = (gid == 0) ? -1 : facetToOctnode_i[gid - 1];
			int rightNeighbor = (gid == numLines - 1) ? -1 : facetToOctnode_i[gid + 1];
			int me = facetToOctnode_i[gid];
			//If my left neighbor doesn't go to the same octnode I go to
			if (leftNeighbor != me) {
				//Then I am the first LCP/Facet belonging to my octnode
				facetPairs_o[me].first = gid;
			}
			//If my right neighbor doesn't go the the same octnode I go to
			if (rightNeighbor != me) {
				//Then I am the last LCP/Facet belonging to my octnode
				facetPairs_o[me].last = gid;
			}
		}
		return CL_SUCCESS;
	}
#endif

	// Get Quadrant (probably should be moved)
#ifndef OpenCL
	inline unsigned char getQuadrant(BigUnsigned *lcp, unsigned char lcpShift, unsigned char i) {
		BigUnsigned buMask, result;
		cl_int quadrantMask = (DIM == 2) ? 3 : 7;
		initBlkBU(&buMask, quadrantMask);

		shiftBULeft(&buMask, &buMask, i * DIM + lcpShift);
		andBU(&result, &buMask, lcp);
		shiftBURight(&result, &result, i * DIM + lcpShift);
		return (result.len == 0) ? 0 : result.blk[0];
	}
#endif

	//Get Node Center From LCP (probably should be moved)
#ifndef OpenCL
	inline floatn getNodeCenterFromLCP(BigUnsigned *LCP, cl_int LCPLength, float octreeWidth) {
		cl_int level = LCPLength / DIM;
		cl_int lcpShift = LCPLength % DIM;
		floatn center = { 0.0, 0.0 };
		float centerShift = octreeWidth / (2.0 * (1 << level));

		for (int i = 0; i < level; ++i) {
			unsigned quadrant = getQuadrant(LCP, lcpShift, i);
			center.x += (quadrant & (1 << 0)) ? centerShift : -centerShift;
			center.y += (quadrant & (1 << 1)) ? centerShift : -centerShift;
#if DIM == 3
			center.z += (quadrant & (1 << 2)) ? centerShift : -centerShift;
#endif
			centerShift *= 2.0;
		}
		return center;
	}
#endif

	// Find Conflict Cells
#ifdef OpenCL
	__kernel void FindConflictCellsKernel(
		__global OctNode *octree,
		__global Leaf *leaves,
		__global int* nodeToFacet,
		__global Pair *facetBounds,
		__global Line* lines,
		cl_int numLines,
		__global intn* qpoints,
		cl_int qwidth,
		__global Conflict* conflicts
	) {
		const int gid = get_global_id(0);
		FindConflictCells(
			gid, octree, leaves, nodeToFacet, facetBounds, 
			lines, numLines, qpoints, qwidth, conflicts);
	}
#else
	inline cl_int FindConflictCells_p(
		cl::Buffer octree_i,
		cl::Buffer leaves_i,
		cl_int numLeaves,
		cl::Buffer LCPToLine_i,
		cl::Buffer LCPBounds_i,
		cl::Buffer lines_i,
		cl_int numLines,
		cl::Buffer &qpoints_i,
		cl_int qwidth,
		cl::Buffer &conflicts_o
	) {
		startBenchmark();
		//Two lines are required for an ambigous cell to appear.
		if (numLines < 2) return CL_INVALID_ARG_SIZE;
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl::Kernel &kernel = CLFW::Kernels["FindConflictCellsKernel"];
		cl::Buffer initialConflictsBuffer;
		cl_int error = 0;
		bool isOld;
//		Conflict initialConflict;
//		initialConflict.color = initialConflict.q1[0] = initialConflict.q1[1] = initialConflict.q2[0] = initialConflict.q2[1] = -1;

		/* We need to initialize the conflict info to -1, but only initialize if we're forced to do so. */
		error |= CLFW::get(conflicts_o, "sparseConflicts", nextPow2(numLeaves) * sizeof(Conflict));
		//error |= CLFW::get(initialConflictsBuffer, "initialConflicts", nextPow2(numLeaves) * sizeof(Conflict), isOld);
		//if (!isOld) error |= queue.enqueueFillBuffer<Conflict>(initialConflictsBuffer, { initialConflict }, 0, nextPow2(numLeaves) * sizeof(Conflict));
		//error |= queue.enqueueCopyBuffer(initialConflictsBuffer, conflicts_o, 0, 0, nextPow2(numLeaves) * sizeof(Conflict));
		error |= kernel.setArg(0, octree_i);
		error |= kernel.setArg(1, leaves_i);
		error |= kernel.setArg(2, LCPToLine_i);
		error |= kernel.setArg(3, LCPBounds_i);
		error |= kernel.setArg(4, lines_i);
		error |= kernel.setArg(5, numLines);
		error |= kernel.setArg(6, qpoints_i);
		error |= kernel.setArg(7, qwidth);
		error |= kernel.setArg(8, conflicts_o);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numLeaves), cl::NullRange);

		stopBenchmark();
		return error;
	}
	inline cl_int FindConflictCells_s(
		vector<OctNode> &octree_i,
		vector<Leaf> &leaves_i,
		vector<cl_int> &nodeToFacet_i,
		vector<Pair> &facetBounds_i,
		vector<Line> &lines_i,
		vector<intn> &qpoints_i,
		cl_int qwidth,
		vector<Conflict> &conflicts_o
		) {
		startBenchmark();
		conflicts_o.resize(leaves_i.size());

		//Two lines are required for an ambigous cell to appear.
		if (lines_i.size() < 2) return CL_INVALID_ARG_SIZE;

		for (cl_int i = 0; i < leaves_i.size(); ++i) {
			FindConflictCells(
				i, octree_i.data(), leaves_i.data(), nodeToFacet_i.data(), facetBounds_i.data(), 
				lines_i.data(), lines_i.size(), qpoints_i.data(), qwidth, conflicts_o.data());
		}
		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

//	// Sample Conflict Counts
//#ifndef OpenCL
//	inline cl_int SampleConflictCounts_s(cl_int totalOctnodes, Conflict *conflicts, int *totalAdditionalPoints,
//		vector<int> &counts, Line* orderedLines, intn* qPoints, vector<intn> &newPoints) {
//		startBenchmark();
//		*totalAdditionalPoints = 0;
//		//This is inefficient. We should only iterate over the conflict leaves, not all leaves. (reduce to find total conflicts)
//		for (int i = 0; i < totalOctnodes * 4; ++i) {
//			Conflict c = conflicts[i];
//			int currentTotalPoints = 0;
//
//			if (conflicts[i].color == -2)
//			{
//				ConflictInfo info;
//				Line firstLine = orderedLines[c.q1[0]];
//				Line secondLine = orderedLines[c.q1[1]];
//				intn q1 = qPoints[firstLine.first];
//				intn q2 = qPoints[firstLine.second];
//				intn r1 = qPoints[secondLine.first];
//				intn r2 = qPoints[secondLine.second];
//				sample_conflict_count(&info, q1, q2, r1, r2, c.origin, c.width);
//
//				const int n = info.num_samples;
//				for (int i = 0; i < info.num_samples; ++i) {
//					floatn sample;
//					sample_conflict_kernel(i, &info, &sample);
//					newPoints.push_back(convert_intn(sample));
//				}
//
//				*totalAdditionalPoints += n;
//
//				////Bug here...
//				//if (currentTotalPoints == 0) {
//				//    printf("Origin: %d %d Width %d (%d %d) (%d %d) : (%d %d) (%d %d) \n", conflicts[i].origin.x, conflicts[i].origin.y,
//				//        conflicts[i].width, qPoints[firstLine.first].x, qPoints[firstLine.first].y,
//				//        qPoints[firstLine.second].x, qPoints[firstLine.second].y,
//				//        qPoints[secondLine.first].x, qPoints[secondLine.first].y,
//				//        qPoints[secondLine.second].x, qPoints[secondLine.second].y);
//				//}
//			}
//			counts[i] = currentTotalPoints;
//			//    cl_int color;
//			//cl_int i[2];
//			//cl_float i2[2];
//			//cl_int width;
//			//intn origin;
//		}
//		stopBenchmark();
//		return CL_SUCCESS;
//	}
//#endif

	// Get Resolution Points Info
#ifdef OpenCL
	__kernel void CountResolutionPointsKernel(
		__global Conflict* conflicts,
		__global intn* qPoints,
		__global ConflictInfo* info_array,
		__global int* resolutionCounts
	)
	{
		const int gid = get_global_id(0);
//    if (gid == 0) {
//      printf("gpu size of conflict info: %d\n", sizeof(ConflictInfo));
//    }
		Conflict c = conflicts[gid];
		ConflictInfo info = { 0 };
    
		// debug
		bool debug = false;//(gid > 1 && gid < 10);
		info.line_pairs[0].s0 = debug ? 1 : 0;
		
		intn q1 = qPoints[c.q1[0]];
		intn q2 = qPoints[c.q1[1]];
		intn r1 = qPoints[c.q2[0]];
		intn r2 = qPoints[c.q2[1]];
    
		if (debug) {
			printf("gpu\n");
//      printf("gpu: q1: (%d, %d)\n", q1.x, q1.y);
//      printf("gpu: q2: (%d, %d)\n", q2.x, q2.y);
//      printf("gpu: r1: (%d, %d)\n", r1.x, r1.y);
//      printf("gpu: r2: (%d, %d)\n", r2.x, r2.y);
//      printf("gpu: o: (%d, %d)\n", c.origin.x, c.origin.y);
//      printf("gpu: w: %d\n", c.width);
		}
		sample_conflict_count(&info, q1, q2, r1, r2, c.origin, c.width);
		if (debug) {
			printf("%d gpu - info.num_samples = %d\n", gid, info.num_samples);
		}

		info_array[gid] = info;
		resolutionCounts[gid] = info.num_samples;
	}
#else
	inline cl_int GetResolutionPointsInfo_p(
		cl::Buffer &conflicts_i, 
		cl_int numConflicts, 
		cl::Buffer &qpoints_i,
		cl::Buffer &conflictInfo_o, 
		cl::Buffer &numPtsPerConflict_o) 
	{
		cl_int error = 0;
		startBenchmark();
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl::Kernel &kernel = CLFW::Kernels["CountResolutionPointsKernel"];

		int globalSize = nextPow2(numConflicts);
//    cout<<"cpu sizeof conflict info: "<<sizeof(ConflictInfo)<<endl;
		error |= CLFW::get(conflictInfo_o, "conflictInfoBuffer", globalSize * sizeof(ConflictInfo));
		error |= CLFW::get(numPtsPerConflict_o, "numPtsPerConflict", globalSize * sizeof(cl_int));
		
		error |= kernel.setArg(0, conflicts_i);
		error |= kernel.setArg(1, qpoints_i);
		error |= kernel.setArg(2, conflictInfo_o);
		error |= kernel.setArg(3, numPtsPerConflict_o);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numConflicts), cl::NullRange);
		stopBenchmark();
		return error;
	}
  
	inline cl_int GetResolutionPointsInfo_s(
		vector<Conflict> &conflicts_i,
		vector<intn> &qpoints_i,
		vector<ConflictInfo> &conflictInfo_o,
		vector<cl_int> &numPtsPerConflict_o) {
		conflictInfo_o.resize(conflicts_i.size());
		numPtsPerConflict_o.resize(conflicts_i.size());
		startBenchmark();
		for (int gid = 0; gid < conflicts_i.size(); ++gid) {
			Conflict c = conflicts_i[gid];
			ConflictInfo info = { 0 };
      
			bool debug = false;//(gid > 1 && gid < 10);
			info.line_pairs[0].s0 = debug ? 1 : 0;
			
			intn q1 = qpoints_i[c.q1[0]];
			intn q2 = qpoints_i[c.q1[1]];
			intn r1 = qpoints_i[c.q2[0]];
			intn r2 = qpoints_i[c.q2[1]];
			if (debug) {
				printf("cpu\n");
//        printf("cpu: q1: (%d, %d)\n", q1.x, q1.y);
//        printf("cpu: q2: (%d, %d)\n", q2.x, q2.y);
//        printf("cpu: r1: (%d, %d)\n", r1.x, r1.y);
//        printf("cpu: r2: (%d, %d)\n", r2.x, r2.y);
//        printf("cpu: o: (%d, %d)\n", c.origin.x, c.origin.y);
//        printf("cpu: w: %d\n", c.width);
			}
      
			sample_conflict_count(&info, q1, q2, r1, r2, c.origin, c.width);
			if (debug) {
				printf("%d cpu - info.num_samples = %d\n", gid, info.num_samples);
			}

			conflictInfo_o[gid] = info;
			numPtsPerConflict_o[gid] = info.num_samples;
		}

		stopBenchmark();
		return CL_SUCCESS;
	}
#endif

	//Predicate Conflict To Point
#ifdef OpenCL
	__kernel void PredicatePointToConflictKernel(
		__global cl_int* scannedNumPtsPerConflict, 
		__global cl_int* predicates
		) {
		predPntToConflict(scannedNumPtsPerConflict, predicates, get_global_id(0));
	}
#else
	inline cl_int PredicatePointToConflict_p(cl::Buffer &scannedNumPtsPerConflict_i, cl_int numConflicts, cl_int numResPts, cl::Buffer &predication_o) {
		cl_int error = 0;
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl::Kernel &kernel = CLFW::Kernels["PredicatePointToConflictKernel"];

		bool isOld;
		cl::Buffer zeroBuffer;
		error |= CLFW::get(zeroBuffer, "pPntToConfZero", sizeof(cl_int) * nextPow2(numResPts), isOld);
		error |= CLFW::get(predication_o, "pPntToConfl", sizeof(cl_int) * nextPow2(numResPts));
		if (!isOld) error |= queue.enqueueFillBuffer<cl_int>(zeroBuffer, { 0 }, 0, sizeof(cl_int) * nextPow2(numResPts));
		error |= queue.enqueueCopyBuffer(zeroBuffer, predication_o, 0, 0, sizeof(cl_int) * nextPow2(numResPts));
		error |= kernel.setArg(0, scannedNumPtsPerConflict_i);
		error |= kernel.setArg(1, predication_o);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numConflicts - 1), cl::NullRange);
		
		return error;
	}

	inline cl_int PredicatePointToConflict_s(vector<cl_int> scannedNumPtsPerConflict_i, cl_int numResPts, vector<cl_int> &predication_o) {
		predication_o.resize(numResPts, 0);
		for (int i = 0; i < scannedNumPtsPerConflict_i.size() - 1; ++i) {
			predPntToConflict(scannedNumPtsPerConflict_i.data(), predication_o.data(), i);
		}
		return CL_SUCCESS;
	}
#endif

	// Get Resolution Points
#ifdef OpenCL
	__kernel void GetResolutionPointsKernel(
		__global Conflict* conflicts_i,
		__global ConflictInfo* conflictInfo_i,
		__global cl_int* scannedNumPtsPerConflict_i,
		__global cl_int* pntToConflict_i,
		__global intn* qpoints_i,
		__global intn* resolutionPoints_o
	)
	{
		const int gid = get_global_id(0);
		cl_int pntToConflict = pntToConflict_i[gid];
		Conflict c = conflicts_i[pntToConflict];
		ConflictInfo info = conflictInfo_i[pntToConflict];

		intn q1 = qpoints_i[c.q1[0]];
		intn q2 = qpoints_i[c.q1[1]];
		intn r1 = qpoints_i[c.q2[0]];
		intn r2 = qpoints_i[c.q2[1]];
		cl_int totalPrevPts = (pntToConflict == 0) ? 0 : scannedNumPtsPerConflict_i[pntToConflict - 1];
		cl_int localIndx = gid - totalPrevPts;
		floatn sample;
		sample_conflict_kernel(localIndx, &info, &sample);
		resolutionPoints_o[gid] = convert_intn(sample);
	}
#else
	inline cl_int GetResolutionPoints_p(
		cl::Buffer &conflicts_i,
		cl::Buffer &conflictInfo_i,
		cl::Buffer &scannedNumPtsPerConflict_i,
		cl_int numResPts,
		cl::Buffer &pntToConflict_i,
		cl::Buffer &qpoints_i,
		cl::Buffer &resolutionPoints_o) 
	{
		startBenchmark();
		cl::CommandQueue &queue = CLFW::DefaultQueue;
		cl::Kernel &kernel = CLFW::Kernels["GetResolutionPointsKernel"];
		cl_int error = 0;

		error |= CLFW::get(resolutionPoints_o, "ResPts", nextPow2(numResPts) * sizeof(intn));
		error |= kernel.setArg(0, conflicts_i);
		error |= kernel.setArg(1, conflictInfo_i);
		error |= kernel.setArg(2, scannedNumPtsPerConflict_i);
		error |= kernel.setArg(3, pntToConflict_i);
		error |= kernel.setArg(4, qpoints_i);
		error |= kernel.setArg(5, resolutionPoints_o);
		error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numResPts), cl::NullRange);
		stopBenchmark();
		return error;
	}

	inline cl_int GetResolutionPoints_s(
		vector<Conflict> &conflicts_i, 
		vector<ConflictInfo> &conflictInfo_i,
		vector<cl_int> &scannedNumPtsPerConflict_i,
		vector<cl_int> &pntToConflict_i, 
		cl_int numResPts,
		vector<intn> &qpoints_i, 
		vector<intn> &resolutionPoints_o
		) {
		resolutionPoints_o.resize(numResPts);
		for (int gid = 0; gid < numResPts; gid++) {
			cl_int pntToConflict = pntToConflict_i[gid];
			Conflict c = conflicts_i[pntToConflict];
			ConflictInfo info = conflictInfo_i[pntToConflict];
			
			intn q1 = qpoints_i[c.q1[0]];
			intn q2 = qpoints_i[c.q1[1]];
			intn r1 = qpoints_i[c.q2[0]];
			intn r2 = qpoints_i[c.q2[1]];
			cl_int totalPrevPts = (pntToConflict == 0) ? 0 : scannedNumPtsPerConflict_i[pntToConflict - 1];
			cl_int localIndx = gid - totalPrevPts;
			floatn sample;
			sample_conflict_kernel(localIndx, &info, &sample);
			resolutionPoints_o[gid] = convert_intn(sample);
		}
		return CL_SUCCESS;
	}
#endif

#pragma endregion

#ifndef OpenCL
}
#endif

#undef benchmark
