///*******************************************************
// ** Generalized Voronoi Diagram Project               **
// ** Copyright (c) 2015 John Martin Edwards            **
// ** Scientific Computing and Imaging Institute        **
// ** 72 S Central Campus Drive, Room 3750              **
// ** Salt Lake City, UT 84112                          **
// **                                                   **
// ** For information about this project contact        **
// ** John Edwards at                                   **
// **    edwardsjohnmartin@gmail.com                    **
// ** or visit                                          **
// **    sci.utah.edu/~jedwards/research/gvd/index.html **
// *******************************************************/

#include <cstdlib>
#include <fstream>
#include "./Options.h"

//------------------------------------------------------------------------------
// Options
//------------------------------------------------------------------------------
namespace Options {
	int computeDevice = -1;
	std::vector<std::string> filenames;
	unsigned char max_level = 6;
	bool series = false; //Not implemented

	float bbxmin = -1;
	float bbymin = -1;
	float bbxmax = -1;
	float bbymax = -1;
  
// Render settings
	bool showObjectVertices = false;
	bool showObjects = true;
	bool showQuantizedObjects = false;
	bool showOctree = true;
	bool showObjectIntersections = false;
	bool showResolutionPoints = false;
	bool showInstructions = true;

	bool pruneOctree = true;

	float conflict_color[3] = { 1.0, 0.0, 0.0 };

	int maxConflictIterations = 3;

	bool zoomMode;

	bool debug = true;
	bool benchmarking = false;
	std::string cl_options = "-cl-std=CL2.0 -I ./Sources -D OpenCL -cl-mad-enable";//"-x spir -cl-std=CL2.0";

#define $(flag) (strcmp(argv[i], flag) == 0)
	bool processArg(int& i, char** argv) {
	int orig_i = i;
	if $("-l") {
		++i;
		max_level = atoi(argv[i]);
		++i;
	}
	else if $("-d") {
		printf("Setting device\n");
		++i;
		computeDevice = atoi(argv[i]);
		++i;
	}
	else if $("-m") {
		++i;
		maxConflictIterations = atoi(argv[i]);
		++i;
	}
	else if $("-v") {
		++i;
		debug = true;
	}
	else if $("--benchmarking") {
		++i;
		benchmarking = true;
	}
	else if $("-s") {
		++i;
		series = true;
	}
	else if $("-o") {
		++i;
		cl_options += "-cl-opt-disable ";
	}
	else if $("--disable-conflict-color") {
		++i;
		conflict_color[1] = 1.0;
		conflict_color[2] = 1.0;
	}
	else if $("-bb") {
		++i;
		bbxmin = atof(argv[i]);
		++i;
		bbymin = atof(argv[i]);
		++i;
		bbxmax = atof(argv[i]);
		++i;
		bbymax = atof(argv[i]);
		++i;
	}
		else if $("--show-respts") {
			++i;
			showResolutionPoints = true;
		}
	else if $("-f") {
		++i;
		std::ifstream in(argv[i]);
		std::string f;
		std::getline(in, f);
		while (in && !f.empty()) {
		if (f[0] != '#') {
			filenames.push_back(f);
		}
		std::getline(in, f);
		}
		++i;
		// } else if (strcmp(argv[i], "-a") == 0) {
		//   ++i;
		//   o.ambiguous_max_level = atoi(argv[i]);
		//   ++i;
		// } else if (strcmp(argv[i], "-x") == 0) {
		//   ++i;
		//   o.test_axis = 0;
		// } else if (strcmp(argv[i], "-y") == 0) {
		//   ++i;
		//   o.test_axis = 1;
		// } else if (strcmp(argv[i], "-z") == 0) {
		//   ++i;
		//   o.test_axis = 2;
		// } else if (strcmp(argv[i], "--tl") == 0) {
		//   ++i;
		//   o.test_num = atoi(argv[i]);
		//   ++i;
		// } else if (strcmp(argv[i], "--no-buffer") == 0) {
		//   o.make_buffer = false;
		//   ++i;
		// } else if (strcmp(argv[i], "-h") == 0) {
		//   o.help = true;
		//   ++i;
		// } else if (strcmp(argv[i], "--gpu") == 0) {
		//   o.gpu = true;
		//   ++i;
		// } else if (strcmp(argv[i], "--cpu") == 0) {
		//   o.gpu = false;
		//   ++i;
		// } else if (strcmp(argv[i], "--opencl-log") == 0) {
		//   o.opencl_log = true;
		//   ++i;
		// } else if (strcmp(argv[i], "--cell-of-interest") == 0) {
		//   ++i;
		//   o.cell_of_interest = atoi(argv[i]);
		//   ++i;
		// } else if (strcmp(argv[i], "--level-of-interest") == 0) {
		//   ++i;
		//   o.level_of_interest = atoi(argv[i]);
		//   ++i;
		// } else if (strcmp(argv[i], "--karras") == 0) {
		//   ++i;
		//   o.karras_iterations = atoi(argv[i]);
		//   ++i;
		// } else if (strcmp(argv[i], "--test") == 0) {
		//   ++i;
		//   o.test = atoi(argv[i]);
		//   ++i;
		// } else if (strcmp(argv[i], "--bb-scale") == 0) {
		//   ++i;
		//   o.bb_scale = atof(argv[i]);
		//   ++i;
		// } else if (strcmp(argv[i], "--rs") == 0) {
		//   o.restricted_surface = true;
		//   ++i;
		// } else if (strcmp(argv[i], "--center") == 0) {
		//   ++i;
		//   o.center = atoi(argv[i]);
		//   ++i;
		// } else if (strcmp(argv[i], "--showObjectVertices") == 0) {
		//   ++i;
		//   o.showObjectVertices = true;
		// } else if (strcmp(argv[i], "--hideObjects") == 0) {
		//   ++i;
		//   o.showObjects = false;
		// } else if (strcmp(argv[i], "--jitter") == 0) {
		//   ++i;
		//   o.jitter = true;
		// } else if (strcmp(argv[i], "--showOctree") == 0) {
		//   ++i;
		//   o.showOctree = true;
	}
	return i != orig_i;
	}
#undef $
}
