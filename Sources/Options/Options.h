/*******************************************************
 ** Generalized Voronoi Diagram Project               **
 ** Copyright (c) 2015 John Martin Edwards            **
 ** Scientific Computing and Imaging Institute        **
 ** 72 S Central Campus Drive, Room 3750              **
 ** Salt Lake City, UT 84112                          **
 **                                                   **
 ** For information about this project contact        **
 ** John Edwards at                                   **
 **    edwardsjohnmartin@gmail.com                    **
 ** or visit                                          **
 **    sci.utah.edu/~jedwards/research/gvd/index.html **
 *******************************************************/

#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>

//static const unsigned char kMaxLevel = 24;

//------------------------------------------------------------------------------
// Options
//------------------------------------------------------------------------------
namespace Options {
extern int device;
extern std::vector<std::string> filenames;
  extern unsigned char max_level;
  //int tri_threshold;
  //bool simple_dist;
  //bool timings;
  //int ambiguous_max_level;
  //bool simple_q;
  //bool full_subdivide;
  //bool make_buffer;
  //bool report_statistics;
  //bool gpu;
  //bool opencl_log;
  //int cell_of_interest;
  //int level_of_interest;
  //float bb_scale;
  //std::set<int> cells_of_interest;
  //int center;
  //bool restricted_surface;
  //int verts_alloc_factor;
  //int karras_iterations;
  //int test;
  //int test_num;
  //int test_axis;
  //bool help;
  //std::map<std::string, std::string> key2value;

  // Render settings
  extern bool showObjectVertices;
  extern bool showObjects;
  extern bool showOctree;
  extern bool showSketcher;

  extern bool zoomMode;
  //bool jitter;

bool processArg(int& i, char** argv);

// public:
//  Options()
//      : max_level(kMaxLevel),
//      tri_threshold(1), simple_dist(true), timings(true),
//        ambiguous_max_level(0), test(-1), showObjectVertices(true),
//        showObjects(true), jitter(false),
//        showOctree(true), test_num(0), test_axis(0) {
//    ReadOptionsFile();
//  }
//
  //Options(unsigned char max_level_,
  //              int tri_threshold_, bool simple_dist_, bool timings_,
  //              bool report_statistics_)
  //    : max_level(max_level_), tri_threshold(tri_threshold_),
  //      simple_dist(simple_dist_), timings(timings_),
  //      ambiguous_max_level(max_level_), simple_q(false),
  //      full_subdivide(false), make_buffer(true),
  //      report_statistics(report_statistics_),
  //      gpu(false),
  //      opencl_log(false), cell_of_interest(-1), level_of_interest(-1),
  //      bb_scale(1), center(-1),
  //      restricted_surface(false),
  //      verts_alloc_factor(3), karras_iterations(1), test(-1),
  //      help(false), test_num(2), test_axis(0) {
  //      ReadOptionsFile();
  //}
//
//  static Options For2D() {
//    Options o(kMaxLevel,
//                    0, false, false, false);
//    o.gpu = false;
//    return o;
//  }
//  static Options For3D() {
//    Options o(kMaxLevel,
//                    5, false, true, true);
//#ifdef __OPEN_CL_SUPPORT__
//    o.gpu = true;
//#else
//    o.gpu = false;
//#endif
//    return o;
//  }
//
//  bool ProcessArg(int& i, char** argv);
//
//  bool OfInterest(int vi) const {
//    if (cell_of_interest == -1) return true;
//    return cells_of_interest.find(vi) != cells_of_interest.end();
//  }
//
//  std::string Value(
//    const std::string& key, const std::string& default_value) const;
//  bool BoolValue(const std::string& key, const bool default_value) const;
//  int IntValue(const std::string& key, const int default_value) const;
//
//  // The options file is key/value pairs, such as
//  //   TEST_AMBIGUOUS_GPU 1
//  //   DISPLAY_SOMETHING 0
//  //   IMPORTANT_MATRIX 1 0 0 0 1 0 0 0 1
//  void ReadOptionsFile();
};
