#include "catch.hpp"
#include "Octree/Octree2.h"
#include "Kernels/Kernels.h"
#include "HelperFunctions.hpp"

/* Reduction Kernels */
Benchmark("Additive Reduction", "[reduction]") {
	TODO("Benchmark this");
}

/* Predication Kernels */
Benchmark("Predicate by bit", "[predication]") {
	TODO("Benchmark this");
}
Benchmark("Predicate BigUnsigned by bit", "[predication]") {
	TODO("Benchmark this");
}
Benchmark("Predicate Conflict", "[conflict][predication]") {
	TODO("Benchmark this");
}

/* Compaction Kernels */
Benchmark("Integer Compaction", "[compaction]") {
	TODO("Benchmark this");
}
Benchmark("Big Unsigned Compaction", "[compaction]") {
	TODO("Benchmark this");
}
Benchmark("Conflict Compaction", "[conflict][compaction]") {
	TODO("Benchmark this");
}

/* Scan Kernels */
Benchmark("Inclusive Summation Scan", "[scan]") {
	TODO("Benchmark this");
}

/* Sort Routines */
Benchmark("Parallel Radix Sort (Pairs by Key)", "[sort][integration][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BigUnsigned> zpoints = readFromFile<BigUnsigned>("BenchmarkData//binaries//zpoints.bin", numPts);
	vector<cl_int> pointColors = readFromFile<cl_int>("BenchmarkData//binaries//pointColors.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_zpoints_copy, b_pntCols, b_pntCols_copy;
	CLFW::get(b_zpoints, "zpoints", numPts * sizeof(BigUnsigned));
	CLFW::get(b_zpoints_copy, "zpointsc", numPts * sizeof(BigUnsigned));
	CLFW::Upload<BigUnsigned>(zpoints, b_zpoints);
	CLFW::get(b_pntCols, "colors", numPts * sizeof(cl_int));
	CLFW::get(b_pntCols_copy, "colorsc", numPts * sizeof(cl_int));
	CLFW::Upload<cl_int>(pointColors, b_pntCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Radix Sort (Pairs by Key)"){
			CLFW::DefaultQueue.enqueueCopyBuffer(b_zpoints, b_zpoints_copy, 0, 0, numPts * sizeof(BigUnsigned));
			CLFW::DefaultQueue.enqueueCopyBuffer(b_pntCols, b_pntCols_copy, 0, 0, numPts * sizeof(cl_int));
			BEGIN_BENCHMARK
			RadixSortBUIntPairsByKey(b_zpoints_copy, b_pntCols_copy, resln.mbits, numPts);
			END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Parallel Radix Sort (Big Unsigneds)", "[sort][integration][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BigUnsigned> zpoints = readFromFile<BigUnsigned>("BenchmarkData//binaries//zpoints.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_zpoints_copy;
	CLFW::get(b_zpoints, "zpoints", numPts * sizeof(BigUnsigned));
	CLFW::get(b_zpoints_copy, "zpointsc", numPts * sizeof(BigUnsigned));
	CLFW::Upload<BigUnsigned>(zpoints, b_zpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("Parallel Radix Sort (Big Unsigneds)"){
			CLFW::DefaultQueue.enqueueCopyBuffer(b_zpoints, b_zpoints_copy, 0, 0, numPts * sizeof(BigUnsigned));
			BEGIN_BENCHMARK
			RadixSortBigUnsigned_p(b_zpoints_copy, numPts, resln.mbits, to_string(i));
			END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Parallel Radix Sort (BU-Int Pairs by Key)", "[sort][integration]") {
	TODO("Benchmark this");
}

/* Z-Order Kernels*/
Benchmark("Quantize Points", "[zorder][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	vector<floatn> points = readFromFile<floatn>("BenchmarkData//binaries//points.bin", numPts);
	BoundingBox bb = readFromFile<BoundingBox>("BenchmarkData//binaries//bb.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");

	/* Upload dependencies */
	cl::Buffer b_points, b_qpoints;
	CLFW::get(b_points, "points", numPts * sizeof(floatn));
	CLFW::Upload<floatn>(points, b_points);

	/* Benchmark */
	BEGIN_ITERATIONS("Quantize Points"){
		BEGIN_BENCHMARK
		QuantizePoints_p(b_points, numPts, bb, resln.width, to_string(i), b_qpoints);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("QPoints to ZPoints", "[zorder][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<intn> qpoints = readFromFile<intn>("BenchmarkData//binaries//qpoints.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_qpoints, b_zpoints;
	CLFW::get(b_qpoints, "qpoints", numPts * sizeof(intn));
	CLFW::Upload<intn>(qpoints, b_qpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("QPoints to ZPoints"){
		BEGIN_BENCHMARK
		QPointsToZPoints_p(b_qpoints, numPts, resln.bits, to_string(i), b_zpoints);
		END_BENCHMARK
	} END_ITERATIONS;
}

/* Unique Kernels */
Benchmark("Unique Sorted BigUnsigned", "[sort][unique][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BigUnsigned> zpoints = readFromFile<BigUnsigned>("BenchmarkData//binaries//sorted_zpoints.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_zpoints_copy;
	CLFW::get(b_zpoints, "zpoints", numPts * sizeof(BigUnsigned));
	CLFW::get(b_zpoints_copy, "zpointsc", numPts * sizeof(BigUnsigned));
	CLFW::Upload<BigUnsigned>(zpoints, b_zpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("Unique Sorted BigUnsigned"){
		CLFW::DefaultQueue.enqueueCopyBuffer(b_zpoints, b_zpoints_copy, 0, 0, numPts * sizeof(BigUnsigned));
		cl_int uniqueNumPts;
		BEGIN_BENCHMARK
		UniqueSorted(b_zpoints_copy, numPts, to_string(i), uniqueNumPts);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Unique Sorted BigUnsigned color pairs", "[sort][unique][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BigUnsigned> zpoints = readFromFile<BigUnsigned>("BenchmarkData//binaries//sorted_zpoints.bin", numPts);
	vector<cl_int> pntColors = readFromFile<cl_int>("BenchmarkData//binaries//sorted_pointColors.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_zpoints_copy, b_pntCols, b_pntCols_copy;
	CLFW::get(b_zpoints, "zpoints", numPts * sizeof(BigUnsigned));
	CLFW::get(b_zpoints_copy, "zpointsc", numPts * sizeof(BigUnsigned));
	CLFW::Upload<BigUnsigned>(zpoints, b_zpoints);
	CLFW::get(b_pntCols, "colors", numPts* sizeof(cl_int));
	CLFW::get(b_pntCols_copy, "colorsc", numPts* sizeof(cl_int));
	CLFW::Upload<cl_int>(pntColors, b_pntCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Unique Sorted BigUnsigned color pairs"){
		CLFW::DefaultQueue.enqueueCopyBuffer(b_zpoints, b_zpoints_copy, 0, 0, numPts * sizeof(BigUnsigned));
		CLFW::DefaultQueue.enqueueCopyBuffer(b_pntCols, b_pntCols_copy, 0, 0, numPts * sizeof(cl_int));
		cl_int uniqueNumPts;
		BEGIN_BENCHMARK
		UniqueSortedBUIntPair(b_zpoints_copy, b_pntCols_copy, numPts, to_string(i), uniqueNumPts);
		END_BENCHMARK
	} END_ITERATIONS;
}

/* Tree Building Kernels */
Benchmark("Build Binary Radix Tree", "[tree][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//unique_numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BigUnsigned> zpoints = readFromFile<BigUnsigned>("BenchmarkData//binaries//unique_sorted_zpoints.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_brt;
	CLFW::get(b_zpoints, "zpoints", numPts * sizeof(BigUnsigned));
	CLFW::Upload<BigUnsigned>(zpoints, b_zpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("Build Binary Radix Tree"){
		BEGIN_BENCHMARK
		BuildBinaryRadixTree_p(b_zpoints, numPts, resln.mbits, to_string(i), b_brt);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Build Colored Binary Radix Tree", "[tree][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//unique_numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BigUnsigned> zpoints = readFromFile<BigUnsigned>("BenchmarkData//binaries//unique_sorted_zpoints.bin", numPts);
	vector<cl_int> pntCols = readFromFile<cl_int>("BenchmarkData//binaries//unique_sorted_pointColors.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_pntCols, b_brt, b_brtCols;
	CLFW::get(b_zpoints, "zpoints", numPts * sizeof(BigUnsigned));
	CLFW::Upload<BigUnsigned>(zpoints, b_zpoints);
	CLFW::get(b_pntCols, "colors", numPts * sizeof(cl_int));
	CLFW::Upload<cl_int>(pntCols, b_pntCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Build Colored Binary Radix Tree"){
		BEGIN_BENCHMARK;
		BuildColoredBinaryRadixTree_p(b_zpoints, b_pntCols, numPts, resln.mbits, to_string(i), b_brt, b_brtCols);
		END_BENCHMARK;
	} END_ITERATIONS;
}
Benchmark("Propagate Brt Colors", "[tree][selected][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//unique_numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BrtNode> brt = readFromFile<BrtNode>("BenchmarkData//binaries//brt.bin", numPts - 1);
	vector<cl_int> brtCols = readFromFile<cl_int>("BenchmarkData//binaries//unpropagated_brtCols.bin", numPts - 1);

	/* Upload dependencies */
	cl::Buffer b_brt, b_brtCols;
	CLFW::get(b_brt, "brt", (numPts - 1) * sizeof(BrtNode));
	CLFW::Upload<BrtNode>(brt, b_brt);
	CLFW::get(b_brtCols, "brtCols", (numPts - 1) * sizeof(cl_int));
	CLFW::Upload<cl_int>(brtCols, b_brtCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Propagate Brt Colors"){
		BEGIN_BENCHMARK;
		PropagateBRTColors_p(b_brt, b_brtCols, numPts - 1, to_string(i));
		END_BENCHMARK;
	} END_ITERATIONS;
}
Benchmark("Build Quadtree", "[tree][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//unique_numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BrtNode> brt = readFromFile<BrtNode>("BenchmarkData//binaries//brt.bin", numPts - 1);

	/* Upload dependencies */
	cl::Buffer b_brt, nullBuffer, b_octree;
	CLFW::get(b_brt, "brt", (numPts - 1) * sizeof(BrtNode));
	CLFW::Upload<BrtNode>(brt, b_brt);

	/* Benchmark */
	BEGIN_ITERATIONS("Build Quadtree"){
		BEGIN_BENCHMARK
		cl_int numOctnodes;
		BinaryRadixToOctree_p(b_brt, false, nullBuffer, numPts, to_string(i), b_octree, numOctnodes);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Build Pruned Quadtree", "[tree][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//unique_numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BrtNode> brt = readFromFile<BrtNode>("BenchmarkData//binaries//brt.bin", numPts - 1);
	vector<cl_int> brtCols = readFromFile<cl_int>("BenchmarkData//binaries//brtCols.bin", numPts - 1);

	/* Upload dependencies */
	cl::Buffer b_brt, b_brtCols, b_octree;
	CLFW::get(b_brt, "brt", (numPts - 1) * sizeof(BrtNode));
	CLFW::Upload<BrtNode>(brt, b_brt);
	CLFW::get(b_brtCols, "brtCols", (numPts - 1) * sizeof(cl_int));
	CLFW::Upload<cl_int>(brtCols, b_brtCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Build Pruned Quadtree"){
		BEGIN_BENCHMARK
		cl_int numOctnodes;
		BinaryRadixToOctree_p(b_brt, true, b_brtCols, numPts, to_string(i), b_octree, numOctnodes);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Generate Leaves", "[tree][done]") {
	/* Load dependencies */
	cl_int numOctnodes = readFromFile<cl_int>("BenchmarkData//binaries//numOctnodes.bin");
	vector<OctNode> octree = readFromFile<OctNode>("BenchmarkData//binaries//octree.bin", numOctnodes);

	/* Upload dependencies */
	cl::Buffer b_octree, b_leaves;
	CLFW::get(b_octree, "octree", numOctnodes * sizeof(OctNode));
	CLFW::Upload<OctNode>(octree, b_octree);

	/* Benchmark */
	BEGIN_ITERATIONS("Generate Leaves"){
		BEGIN_BENCHMARK
		cl_int numLeaves;
		GetLeaves_p(b_octree, numOctnodes, b_leaves, numLeaves);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Generate Pruned Leaves", "[tree][done]") {
	/* Load dependencies */
	cl_int numOctnodes = readFromFile<cl_int>("BenchmarkData//binaries//pruned_numOctnodes.bin");
	vector<OctNode> octree = readFromFile<OctNode>("BenchmarkData//binaries//pruned_octree.bin", numOctnodes);

	/* Upload dependencies */
	cl::Buffer b_octree, b_leaves;
	CLFW::get(b_octree, "octree", numOctnodes * sizeof(OctNode));
	CLFW::Upload<OctNode>(octree, b_octree);

	/* Benchmark */
	BEGIN_ITERATIONS("Generate Pruned Leaves"){
		BEGIN_BENCHMARK
		cl_int numLeaves;
		GetLeaves_p(b_octree, numOctnodes, b_leaves, numLeaves);
		END_BENCHMARK
	} END_ITERATIONS;
}

/* Ambiguous cell detection kernels */
Benchmark("Get LCPs From Lines", "[conflict]") {
	TODO("Benchmark this");
}
Benchmark("Look Up Octnode From LCP", "[conflict]") {
	TODO("Benchmark this");
}
Benchmark("Get Octnode LCP Bounds", "[conflict]") {
	TODO("Benchmark this");
}
Benchmark("Find Conflict Cells", "[conflict]") {
	TODO("Benchmark this");
}

/* Ambiguous cell resolution kernels */
Benchmark("Sample required resolution points", "[resolution]") {
	TODO("Benchmark this");
}
Benchmark("Predicate Conflict To Point", "[predication][resolution]") {
	TODO("Benchmark this");
}
Benchmark("Get resolution points", "[resolution]") {
	TODO("Benchmark this");
}

/* Quadtree */