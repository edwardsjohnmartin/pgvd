#include "catch.hpp"
#include "Octree/Octree2.h"
#include "Kernels/Kernels.h"
#include "HelperFunctions.hpp"

/* Reduction Kernels */
Benchmark("Additive Reduction", "[reduction]") {
	TODO("Benchmark this");
}
Benchmark("Check Order", "[4way][sort][reduction]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	vector<big> zpoints(numPts);
	for (int i = 0; i < numPts; ++i)
		zpoints[i] = makeBig(i);

	/* Upload dependencies */
	cl::Buffer b_zpoints;
	CLFW::getBuffer(b_zpoints, "zpoints", Kernels::nextPow2(numPts) * sizeof(big));
	CLFW::Upload<big>(zpoints, b_zpoints);
	big maxbig = makeMaxBig();
	CLFW::DefaultQueue.enqueueFillBuffer<big>(b_zpoints, { maxbig }, numPts * sizeof(big), 
		(Kernels::nextPow2(numPts) - numPts) * sizeof(big));

	/* Benchmark */
	BEGIN_ITERATIONS("Check Order") {
		BEGIN_BENCHMARK
		cl_int result;
		CheckBigOrder_p(b_zpoints, Kernels::nextPow2(numPts), result);
		END_BENCHMARK
	} END_ITERATIONS;
}

/* Predication Kernels */
Benchmark("Predicate by bit", "[predication]") {
	TODO("Benchmark this");
}
Benchmark("Predicate big by bit", "[predication]") {
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
Benchmark("Parallel Radix Sort", "[sort][integration][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<unsigned long long> zpoints(numPts);
	for (int i = 0; i < numPts; ++i) zpoints[i] = i;

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_zpoints_copy;
	CLFW::getBuffer(b_zpoints, "zpoints", numPts * sizeof(unsigned long long));
	CLFW::getBuffer(b_zpoints_copy, "zpointsc", numPts * sizeof(unsigned long long));
	CLFW::Upload<unsigned long long>(zpoints, b_zpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("Parallel Radix Sort") {
		CLFW::DefaultQueue.enqueueCopyBuffer(b_zpoints, b_zpoints_copy, 0, 0, numPts * sizeof(unsigned long long));
		BEGIN_BENCHMARK
			OldRadixSort_p(b_zpoints_copy, numPts, resln.mbits);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Parallel Radix Sort (Pairs by Key)", "[sort][integration][done][selected]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<big> zpoints = readFromFile<big>("BenchmarkData//binaries//zpoints.bin", numPts);
	vector<cl_int> pointColors = readFromFile<cl_int>("BenchmarkData//binaries//pointColors.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_zpoints_copy, b_pntCols, b_pntCols_copy;
	CLFW::getBuffer(b_zpoints, "zpoints", numPts * sizeof(big));
	CLFW::getBuffer(b_zpoints_copy, "zpointsc", numPts * sizeof(big));
	CLFW::Upload<big>(zpoints, b_zpoints);
	CLFW::getBuffer(b_pntCols, "colors", numPts * sizeof(cl_int));
	CLFW::getBuffer(b_pntCols_copy, "colorsc", numPts * sizeof(cl_int));
	CLFW::Upload<cl_int>(pointColors, b_pntCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Radix Sort (BU Pairs by Key)") {
		CLFW::DefaultQueue.enqueueCopyBuffer(b_zpoints, b_zpoints_copy, 0, 0, numPts * sizeof(big));
		CLFW::DefaultQueue.enqueueCopyBuffer(b_pntCols, b_pntCols_copy, 0, 0, numPts * sizeof(cl_int));
		BEGIN_BENCHMARK
			RadixSortIntToInt_p(b_zpoints_copy, b_pntCols_copy, numPts, resln.mbits, "");
			//OldRadixSortBUIntPairsByKey(b_zpoints_copy, b_pntCols_copy, resln.mbits, numPts);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Parallel Radix Sort (Big Unsigneds)", "[sort][integration][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<big> zpoints = readFromFile<big>("BenchmarkData//binaries//zpoints.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_zpoints_copy;
	CLFW::getBuffer(b_zpoints, "zpoints", numPts * sizeof(big));
	CLFW::getBuffer(b_zpoints_copy, "zpointsc", numPts * sizeof(big));
	CLFW::Upload<big>(zpoints, b_zpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("Parallel Radix Sort (Big Unsigneds)") {
		CLFW::DefaultQueue.enqueueCopyBuffer(b_zpoints, b_zpoints_copy, 0, 0, numPts * sizeof(big));
		BEGIN_BENCHMARK
			RadixSortBig_p(b_zpoints_copy, numPts, resln.mbits, to_string(i));
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
	CLFW::getBuffer(b_points, "points", numPts * sizeof(floatn));
	CLFW::Upload<floatn>(points, b_points);

	/* Benchmark */
	BEGIN_ITERATIONS("Quantize Points") {
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
	CLFW::getBuffer(b_qpoints, "qpoints", numPts * sizeof(intn));
	CLFW::Upload<intn>(qpoints, b_qpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("QPoints to ZPoints") {
		BEGIN_BENCHMARK
			QPointsToZPoints_p(b_qpoints, numPts, resln.bits, to_string(i), b_zpoints);
		END_BENCHMARK
	} END_ITERATIONS;
}

/* Unique Kernels */
Benchmark("Unique Sorted big", "[sort][unique][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<big> zpoints = readFromFile<big>("BenchmarkData//binaries//sorted_zpoints.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_zpoints_copy;
	CLFW::getBuffer(b_zpoints, "zpoints", numPts * sizeof(big));
	CLFW::getBuffer(b_zpoints_copy, "zpointsc", numPts * sizeof(big));
	CLFW::Upload<big>(zpoints, b_zpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("Unique Sorted big") {
		CLFW::DefaultQueue.enqueueCopyBuffer(b_zpoints, b_zpoints_copy, 0, 0, numPts * sizeof(big));
		cl_int uniqueNumPts;
		BEGIN_BENCHMARK
			UniqueSorted(b_zpoints_copy, numPts, to_string(i), uniqueNumPts);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Unique Sorted big color pairs", "[sort][unique][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<big> zpoints = readFromFile<big>("BenchmarkData//binaries//sorted_zpoints.bin", numPts);
	vector<cl_int> pntColors = readFromFile<cl_int>("BenchmarkData//binaries//sorted_pointColors.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_zpoints_copy, b_pntCols, b_pntCols_copy;
	CLFW::getBuffer(b_zpoints, "zpoints", numPts * sizeof(big));
	CLFW::getBuffer(b_zpoints_copy, "zpointsc", numPts * sizeof(big));
	CLFW::Upload<big>(zpoints, b_zpoints);
	CLFW::getBuffer(b_pntCols, "colors", numPts* sizeof(cl_int));
	CLFW::getBuffer(b_pntCols_copy, "colorsc", numPts* sizeof(cl_int));
	CLFW::Upload<cl_int>(pntColors, b_pntCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Unique Sorted big color pairs") {
		CLFW::DefaultQueue.enqueueCopyBuffer(b_zpoints, b_zpoints_copy, 0, 0, numPts * sizeof(big));
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
	vector<big> zpoints = readFromFile<big>("BenchmarkData//binaries//unique_sorted_zpoints.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_brt;
	CLFW::getBuffer(b_zpoints, "zpoints", numPts * sizeof(big));
	CLFW::Upload<big>(zpoints, b_zpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("Build Binary Radix Tree") {
		BEGIN_BENCHMARK
			BuildBinaryRadixTree_p(b_zpoints, numPts, resln.mbits, to_string(i), b_brt);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Build Colored Binary Radix Tree", "[tree][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//unique_numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<big> zpoints = readFromFile<big>("BenchmarkData//binaries//unique_sorted_zpoints.bin", numPts);
	vector<cl_int> pntCols = readFromFile<cl_int>("BenchmarkData//binaries//unique_sorted_pointColors.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_zpoints, b_pntCols, b_brt, b_brtCols;
	CLFW::getBuffer(b_zpoints, "zpoints", numPts * sizeof(big));
	CLFW::Upload<big>(zpoints, b_zpoints);
	CLFW::getBuffer(b_pntCols, "colors", numPts * sizeof(cl_int));
	CLFW::Upload<cl_int>(pntCols, b_pntCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Build Colored Binary Radix Tree") {
		BEGIN_BENCHMARK;
		BuildColoredBinaryRadixTree_p(b_zpoints, b_pntCols, numPts, resln.mbits, to_string(i), b_brt, b_brtCols);
		END_BENCHMARK;
	} END_ITERATIONS;
}
Benchmark("Propagate Brt Colors", "[tree][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//unique_numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BrtNode> brt = readFromFile<BrtNode>("BenchmarkData//binaries//brt.bin", numPts - 1);
	vector<cl_int> brtCols = readFromFile<cl_int>("BenchmarkData//binaries//unpropagated_brtCols.bin", numPts - 1);

	/* Upload dependencies */
	cl::Buffer b_brt, b_brtCols;
	CLFW::getBuffer(b_brt, "brt", (numPts - 1) * sizeof(BrtNode));
	CLFW::Upload<BrtNode>(brt, b_brt);
	CLFW::getBuffer(b_brtCols, "brtCols", (numPts - 1) * sizeof(cl_int));
	CLFW::Upload<cl_int>(brtCols, b_brtCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Propagate Brt Colors") {
		BEGIN_BENCHMARK;
		PropagateBRTColors_p(b_brt, b_brtCols, numPts - 1, to_string(i));
		END_BENCHMARK;
	} END_ITERATIONS;
}
Benchmark("Build Quadtree from BRT", "[tree][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//unique_numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BrtNode> brt = readFromFile<BrtNode>("BenchmarkData//binaries//brt.bin", numPts - 1);

	/* Upload dependencies */
	cl::Buffer b_brt, nullBuffer, b_octree;
	CLFW::getBuffer(b_brt, "brt", (numPts - 1) * sizeof(BrtNode));
	CLFW::Upload<BrtNode>(brt, b_brt);

	/* Benchmark */
	BEGIN_ITERATIONS("Build Quadtree from BRT") {
		BEGIN_BENCHMARK
			cl_int numOctnodes;
		BinaryRadixToOctree_p(b_brt, false, nullBuffer, numPts, to_string(i), b_octree, numOctnodes);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Build Pruned Quadtree from BRT", "[tree][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//unique_numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<BrtNode> brt = readFromFile<BrtNode>("BenchmarkData//binaries//brt.bin", numPts - 1);
	vector<cl_int> brtCols = readFromFile<cl_int>("BenchmarkData//binaries//brtCols.bin", numPts - 1);

	/* Upload dependencies */
	cl::Buffer b_brt, b_brtCols, b_octree;
	CLFW::getBuffer(b_brt, "brt", (numPts - 1) * sizeof(BrtNode));
	CLFW::Upload<BrtNode>(brt, b_brt);
	CLFW::getBuffer(b_brtCols, "brtCols", (numPts - 1) * sizeof(cl_int));
	CLFW::Upload<cl_int>(brtCols, b_brtCols);

	/* Benchmark */
	BEGIN_ITERATIONS("Build Pruned Quadtree from BRT") {
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
	CLFW::getBuffer(b_octree, "octree", numOctnodes * sizeof(OctNode));
	CLFW::Upload<OctNode>(octree, b_octree);

	/* Benchmark */
	BEGIN_ITERATIONS("Generate Leaves") {
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
	CLFW::getBuffer(b_octree, "octree", numOctnodes * sizeof(OctNode));
	CLFW::Upload<OctNode>(octree, b_octree);

	/* Benchmark */
	BEGIN_ITERATIONS("Generate Pruned Leaves") {
		BEGIN_BENCHMARK
			cl_int numLeaves;
		GetLeaves_p(b_octree, numOctnodes, b_leaves, numLeaves);
		END_BENCHMARK
	} END_ITERATIONS;
}

/* Ambiguous cell detection kernels */
Benchmark("Get LCPs From Lines", "[conflict][done]") {
	/* Load dependencies */
	cl_int numLines = readFromFile<cl_int>("BenchmarkData//binaries//numLines.bin");
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<big> zpoints = readFromFile<big>("BenchmarkData//binaries//zpoints.bin", numPts);
	vector<Line> lines = readFromFile<Line>("BenchmarkData//binaries//lines.bin", numLines);

	/* Upload dependencies */
	cl::Buffer b_lines, b_zpoints, b_LineLCPs;
	CLFW::getBuffer(b_lines, "lines", numLines * sizeof(Line));
	CLFW::getBuffer(b_zpoints, "zpoints", numPts * sizeof(OctNode));
	CLFW::Upload<Line>(lines, b_lines);
	CLFW::Upload<big>(zpoints, b_zpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("Get LCPs From Lines") {
		BEGIN_BENCHMARK
			GetLineLCPs_p(b_lines, numLines, b_zpoints, resln.mbits, b_LineLCPs);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Look Up Octnode From LCP", "[conflict][done]") {
	/* Load dependencies */
	cl_int numOctnodes = readFromFile<cl_int>("BenchmarkData//binaries//pruned_numOctnodes.bin");
	vector<OctNode> octree = readFromFile<OctNode>("BenchmarkData//binaries//pruned_octree.bin", numOctnodes);
	cl_int numLines = readFromFile<cl_int>("BenchmarkData//binaries//numLines.bin");
	vector<LCP> lineLCPs = readFromFile<LCP>("BenchmarkData//binaries//lineLCPs.bin", numLines);

	/* Upload dependencies */
	cl::Buffer b_lineLCPs, b_octree, b_LCPToOctNode;
	CLFW::getBuffer(b_lineLCPs, "lineLCPs", numLines * sizeof(LCP));
	CLFW::getBuffer(b_octree, "octree", numOctnodes * sizeof(OctNode));
	CLFW::Upload<LCP>(lineLCPs, b_lineLCPs);
	CLFW::Upload<OctNode>(octree, b_octree);

	/* Benchmark */
	BEGIN_ITERATIONS("Look Up Octnode From LCP") {
		BEGIN_BENCHMARK
			LookUpOctnodeFromLCP_p(b_lineLCPs, numLines, b_octree, b_LCPToOctNode);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Get Octnode LCP Bounds", "[conflict][done]") {
	/* Load dependencies */
	cl_int numOctnodes = readFromFile<cl_int>("BenchmarkData//binaries//numOctnodes.bin");
	cl_int numLines = readFromFile<cl_int>("BenchmarkData//binaries//numLines.bin");
	vector<cl_int> LCPToOctNode = readFromFile<cl_int>("BenchmarkData//binaries//LCPToOctNode.bin", numLines);

	/* Upload dependencies */
	cl::Buffer b_LCPToOctNode, b_LCPBounds;
	CLFW::getBuffer(b_LCPToOctNode, "LCPToOctNode", numLines * sizeof(cl_int));
	CLFW::Upload<cl_int>(LCPToOctNode, b_LCPToOctNode);

	/* Benchmark */
	BEGIN_ITERATIONS("Get Octnode LCP Bounds") {
		BEGIN_BENCHMARK
			GetLCPBounds_p(b_LCPToOctNode, numLines, numOctnodes, b_LCPBounds);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Find Conflict Cells", "[conflict][done]") {
	/* Load dependencies */
	cl_int numOctnodes = readFromFile<cl_int>("BenchmarkData//binaries//pruned_numOctnodes.bin");
	cl_int numLeaves = readFromFile<cl_int>("BenchmarkData//binaries//pruned_numLeaves.bin");
	cl_int numLines = readFromFile<cl_int>("BenchmarkData//binaries//numLines.bin");
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	Resln resln = readFromFile<Resln>("BenchmarkData//binaries//resln.bin");
	vector<OctNode> octree = readFromFile<OctNode>("BenchmarkData//binaries//pruned_octree.bin", numOctnodes);
	vector<Leaf> leaves = readFromFile<Leaf>("BenchmarkData//binaries//pruned_leaves.bin", numLeaves);
	vector<cl_int> lineIndices = readFromFile<cl_int>("BenchmarkData//binaries//lineIndices.bin", numLines);
	vector<Pair> LCPBounds = readFromFile<Pair>("BenchmarkData//binaries//LCPBounds.bin", numOctnodes);
	vector<Line> lines = readFromFile<Line>("BenchmarkData//binaries//lines.bin", numLines);
	vector<intn> qpoints = readFromFile<intn>("BenchmarkData//binaries//qpoints.bin", numPts);

	/* Upload dependencies */
	cl::Buffer b_octree, b_leaves, b_lineIndices, b_LCPBounds, b_lines, b_qpoints, b_sparseConflicts;
	CLFW::getBuffer(b_octree, "octree", numOctnodes * sizeof(OctNode));
	CLFW::getBuffer(b_leaves, "leaves", numLeaves * sizeof(Leaf));
	CLFW::getBuffer(b_lineIndices, "lineIndices", numLines * sizeof(cl_int));
	CLFW::getBuffer(b_LCPBounds, "LCPBounds", numOctnodes * sizeof(Pair));
	CLFW::getBuffer(b_lines, "lines", numLines * sizeof(Line));
	CLFW::getBuffer(b_qpoints, "qpoints", numPts * sizeof(intn));
	CLFW::Upload<OctNode>(octree, b_octree);
	CLFW::Upload<Leaf>(leaves, b_leaves);
	CLFW::Upload<cl_int>(lineIndices, b_lineIndices);
	CLFW::Upload<Pair>(LCPBounds, b_LCPBounds);
	CLFW::Upload<Line>(lines, b_lines);
	CLFW::Upload<intn>(qpoints, b_qpoints);

	/* Benchmark */
	BEGIN_ITERATIONS("Find Conflict Cells") {
		BEGIN_BENCHMARK
			FindConflictCells_p(b_octree, b_leaves, numLeaves, b_lineIndices,
				b_LCPBounds, b_lines, numLines, b_qpoints, resln.width, b_sparseConflicts);
		END_BENCHMARK
	} END_ITERATIONS;
}

/* Ambiguous cell resolution kernels */
Benchmark("Sample required resolution points", "[resolution][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	cl_int numConflicts = readFromFile<cl_int>("BenchmarkData//binaries//numConflicts.bin");
	vector<intn> qpoints = readFromFile<intn>("BenchmarkData//binaries//qpoints.bin", numPts);
	vector<Conflict> conflicts = readFromFile<Conflict>("BenchmarkData//binaries//conflicts.bin", numConflicts);

	/* Upload dependencies */
	cl::Buffer b_conflicts, b_qpoints, b_conflictInfo, b_numPtsPerConflict;
	CLFW::getBuffer(b_qpoints, "qpoints", numPts * sizeof(intn));
	CLFW::Upload<intn>(qpoints, b_qpoints);
	CLFW::getBuffer(b_conflicts, "conflicts", numConflicts * sizeof(Conflict));
	CLFW::Upload<Conflict>(conflicts, b_conflicts);

	/* Benchmark */
	BEGIN_ITERATIONS("Sample required resolution points") {
		BEGIN_BENCHMARK
			GetResolutionPointsInfo_p(b_conflicts, numConflicts, b_qpoints, b_conflictInfo, b_numPtsPerConflict);
		END_BENCHMARK
	} END_ITERATIONS;
}
Benchmark("Predicate Conflict To Point", "[predication][resolution]") {
	TODO("Benchmark this");
}
Benchmark("Get resolution points", "[resolution][done]") {
	/* Load dependencies */
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	cl_int numResPts = readFromFile<cl_int>("BenchmarkData//binaries//numResPts.bin");
	cl_int numConflicts = readFromFile<cl_int>("BenchmarkData//binaries//numConflicts.bin");
	vector<intn> qpoints = readFromFile<intn>("BenchmarkData//binaries//qpoints.bin", numPts);
	vector<Conflict> conflicts = readFromFile<Conflict>("BenchmarkData//binaries//conflicts.bin", numConflicts);
	vector<cl_int> scannedNumPtsPerConflict = readFromFile<cl_int>("BenchmarkData//binaries//scannedNumPtsPerConflict.bin", numConflicts);
	vector<cl_int> pntToConflict = readFromFile<cl_int>("BenchmarkData//binaries//pntToConflict.bin", numResPts);
	vector<ConflictInfo> conflictInfo = readFromFile<ConflictInfo>("BenchmarkData//binaries//conflictInfo.bin", numConflicts);

	/* Upload dependencies */
	cl::Buffer b_conflicts, b_conflictInfo, b_qpoints, b_scannedNumPtsPerConflict, b_pntToConflict, b_resPts;
	CLFW::getBuffer(b_qpoints, "qpoints", numPts * sizeof(intn));
	CLFW::Upload<intn>(qpoints, b_qpoints);
	CLFW::getBuffer(b_conflicts, "conflicts", numConflicts * sizeof(Conflict));
	CLFW::Upload<Conflict>(conflicts, b_conflicts);
	CLFW::getBuffer(b_scannedNumPtsPerConflict, "scannumpts", sizeof(cl_int) * numConflicts);
	CLFW::Upload<cl_int>(scannedNumPtsPerConflict, b_scannedNumPtsPerConflict);
	CLFW::getBuffer(b_pntToConflict, "pntToConflict", sizeof(cl_int) * numResPts);
	CLFW::Upload<cl_int>(pntToConflict, b_pntToConflict);
	CLFW::getBuffer(b_conflictInfo, "conflictInfo", sizeof(ConflictInfo) * numConflicts);
	CLFW::Upload<ConflictInfo>(conflictInfo, b_conflictInfo);

	/* Benchmark */
	BEGIN_ITERATIONS("Get resolution points") {
		BEGIN_BENCHMARK
			GetResolutionPoints_p(b_conflicts, b_conflictInfo, b_scannedNumPtsPerConflict, numResPts, b_pntToConflict, b_qpoints, b_resPts);
		END_BENCHMARK
	} END_ITERATIONS;
}

/* Quadtree */
Benchmark("Build Unpruned Quadtree", "[tree][done][selected]") {
	/* Load dependencies */
	Options::max_level = 24;
	Options::pruneOctree = false;
	Quadtree q;
	glm::mat4 I;
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	vector<floatn> points = readFromFile<floatn>("BenchmarkData//binaries//points.bin", numPts);
	vector<cl_int> pointCols = readFromFile<cl_int>("BenchmarkData//binaries//pointColors.bin", numPts);
	cl_int numLines = readFromFile<cl_int>("BenchmarkData//binaries//numLines.bin");
	vector<Line> lines = readFromFile<Line>("BenchmarkData//binaries//lines.bin", numLines);
	BoundingBox bb = readFromFile<BoundingBox>("BenchmarkData//binaries//bb.bin");

	/* Benchmark */
	BEGIN_ITERATIONS("Build Quadtree") {
		BEGIN_BENCHMARK
			q.build(points, pointCols, lines, bb);
		END_BENCHMARK
	} END_ITERATIONS;
}

/* Quadtree */
Benchmark("Build Pruned Quadtree", "[tree][done][selected]") {
	/* Load dependencies */
	Options::max_level = 24;
	Options::pruneOctree = true;
	Quadtree q;
	glm::mat4 I;
	cl_int numPts = readFromFile<cl_int>("BenchmarkData//binaries//numPts.bin");
	vector<floatn> points = readFromFile<floatn>("BenchmarkData//binaries//points.bin", numPts);
	vector<cl_int> pointCols = readFromFile<cl_int>("BenchmarkData//binaries//pointColors.bin", numPts);
	cl_int numLines = readFromFile<cl_int>("BenchmarkData//binaries//numLines.bin");
	vector<Line> lines = readFromFile<Line>("BenchmarkData//binaries//lines.bin", numLines);
	BoundingBox bb = readFromFile<BoundingBox>("BenchmarkData//binaries//bb.bin");

	/* Benchmark */
	BEGIN_ITERATIONS("Build Pruned Quadtree") {
		BEGIN_BENCHMARK
			q.build(points, pointCols, lines, bb);
		END_BENCHMARK
	} END_ITERATIONS;
}