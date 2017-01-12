#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

#include "../GLUtilities/gl_utils.h"
#include "./Octree2.h"
#include "../Timer/timer.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using namespace std;
using namespace Kernels;

#define benchmark(text) if (Options::benchmarking) { \
  static log4cplus::Logger logger = log4cplus::Logger::getInstance("Octree2.benchmark"); \
  Timer t(logger, text); \
}
#define check(function) {cl_int error = function; assert_cl_error(error);}

Octree2::Octree2() {
  const int n = 4;

  // Octree drawn using instanced, indexed rendered cubes.
  glGenBuffers(1, &positions_vbo);
  glGenBuffers(1, &instance_vbo);
  glGenBuffers(1, &position_indices_vbo);
  glGenVertexArrays(1, &boxProgram_vao);
  glBindVertexArray(boxProgram_vao);
  glEnableVertexAttribArray(Shaders::boxProgram->position_id);
  glEnableVertexAttribArray(Shaders::boxProgram->offset_id);
  glEnableVertexAttribArray(Shaders::boxProgram->scale_id);
  glEnableVertexAttribArray(Shaders::boxProgram->color_id);
  print_gl_error();
  float points[] = {
    -.5, -.5, -.5,  -.5, -.5, +.5,
    +.5, -.5, +.5,  +.5, -.5, -.5,
    -.5, +.5, -.5,  -.5, +.5, +.5,
    +.5, +.5, +.5,  +.5, +.5, -.5,
  };
  unsigned char indices[] = {
    0, 1, 1, 2, 2, 3, 3, 0,
    0, 4, 1, 5, 2, 6, 3, 7,
    4, 5, 5, 6, 6, 7, 7, 4,
  };
  glBindBuffer(GL_ARRAY_BUFFER, positions_vbo);
  print_gl_error();
  glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
  print_gl_error();
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, position_indices_vbo);
  fprintf(stderr, "position_indices_vbo: %d\n", position_indices_vbo);
  print_gl_error();
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  print_gl_error();
  glVertexAttribPointer(Shaders::boxProgram->position_id, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
  print_gl_error();
  glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
  print_gl_error();
  glVertexAttribPointer(Shaders::boxProgram->offset_id, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), 0);
  print_gl_error();
  glVertexAttribPointer(Shaders::boxProgram->scale_id, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
  print_gl_error();
  glVertexAttribPointer(Shaders::boxProgram->color_id, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(4 * sizeof(float)));
  print_gl_error();
  glVertexAttribDivisor(Shaders::boxProgram->offset_id, 1);
  print_gl_error();
  glVertexAttribDivisor(Shaders::boxProgram->scale_id, 1);
  print_gl_error();
  glVertexAttribDivisor(Shaders::boxProgram->color_id, 1);
  glBindVertexArray(0);
  print_gl_error();

  resln = make_resln(1 << Options::max_level);
}

void Octree2::build(const PolyLines *polyLines) {
  using namespace GLUtilities;
  Sketcher::instance()->clear();

  int totalIterations = 0;
  int previousSize;
  octreeSize = 0;
  resolutionPointsSize = 0;

  karras_points.clear();
  gl_instances.clear();
  octree.clear();
  resolutionPoints.clear();

  /* Quantize the polygon points. */
  getPoints(polyLines);
  if (karras_points.size() == 0) return;

  totalPoints = karras_points.size();
  getBoundingBox(totalPoints);
  UploadKarrasPoints(karras_points, karrasPointsBuffer);
  UploadLines(lines, linesBuffer);
  getQuantizedPoints();
  getZOrderPoints();

  do {
    totalIterations++;

    CLFW::DefaultQueue = CLFW::Queues[0];
    previousSize = octreeSize;

    addResolutionPoints();
    getZOrderPoints(); //TODO: only z-order the additional points here...
    /*
      On one queue, sort the lines for the ambiguous cell detection.
      On the other, start building the karras octree.
    */
    CLFW::DefaultQueue = CLFW::Queues[1];
    getUnorderedBCellFacetPairs();
    CLFW::DefaultQueue = CLFW::Queues[0];
    getVertexOctree();
    Kernels::GetLeaves_p(CLFW::Buffers["octree"], leavesBuffer, octreeSize, totalLeaves);
    CLFW::Queues[0].finish();
    CLFW::Queues[1].finish();
    /* Identify the cells that contain more than one object. */
    if (lines.size() == 0 || lines.size() == 1) break;
    getConflictCells();
    /* Resolve one iteration of conflicts. */
    //getResolutionPoints();
    //drawResolutionPoints();
    break;
    if (totalIterations == Options::maxConflictIterations) {
      //LOG4CPLUS_WARN(logger, "*** Warning: breaking out of conflict "
        //<< "detection loop early for debugging purposes.");
      // exit(0);
    break;
    }
  } while (previousSize != octreeSize && resolutionPointsSize != 0);

  cl::Buffer octreeBuffer = CLFW::Buffers["octree"];
  octree.resize(octreeSize);
  check(CLFW::DefaultQueue.enqueueReadBuffer(
    octreeBuffer, CL_TRUE, 0, sizeof(OctNode)*octreeSize, octree.data()));

  /* Add the octnodes and conflict cells so they'll be rendered with OpenGL. */
  addOctreeNodes();
  addConflictCells();
}

inline floatn getMinFloat(const floatn a, const floatn b) {
  floatn result;
  for (int i = 0; i < DIM; ++i) {
    result.s[i] = (a.s[i] < b.s[i]) ? a.s[i] : b.s[i];
  }
  return result;
}

inline floatn getMaxFloat(const floatn a, const floatn b) {
  floatn result;
  for (int i = 0; i < DIM; ++i) {
    result.s[i] = (a.s[i] > b.s[i]) ? a.s[i] : b.s[i];
  }
  return result;
}

void Octree2::getPoints(const PolyLines *polyLines) {
  benchmark("getPoints");

  const vector<vector<floatn>> polygons = polyLines->getPolygons();
  lines = polyLines->getLines();

  // Get all vertices into a 1D array (karras_points).
  for (int i = 0; i < polygons.size(); ++i) {
    const vector<floatn>& polygon = polygons[i];
    for (int j = 0; j < polygon.size(); ++j) {
      karras_points.push_back(polygon[j]);
    }
  }
}

void Octree2::getBoundingBox(const int totalPoints) {
  benchmark("getBoundingBox");

  if (Options::xmin == -1 && Options::xmax == -1) {
    //Probably should be parallelized...
    floatn minimum = karras_points[0];
    floatn maximum = karras_points[0];
    for (int i = 1; i < totalPoints; ++i) {
      minimum = getMinFloat(karras_points[i], minimum);
      maximum = getMaxFloat(karras_points[i], maximum);
    }
    bb = BB_initialize(&minimum, &maximum);
    bb = BB_make_centered_square(&bb);
  }
  else {
    bb.initialized = true;
    bb.minimum = make_floatn(Options::xmin, Options::ymin);
    bb.maximum = make_floatn(Options::xmax, Options::ymax);
    bb.maxwidth = BB_max_size(&bb);
  }
}

void Octree2::getQuantizedPoints(int numResolutionPoints) {
  check(Kernels::QuantizePoints_p(totalPoints, karrasPointsBuffer, quantizedPointsBuffer, bb.minimum, resln.width, bb.maxwidth));
}

void Octree2::getZOrderPoints() {
  static log4cplus::Logger logger =
    log4cplus::Logger::getInstance("Octree2.makeZOrderPoints");

  check(Kernels::PointsToMorton_p( quantizedPointsBuffer, zpoints, totalPoints, resln.bits));
  
  if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
    vector<BigUnsigned> buf;
    Kernels::DownloadZPoints(buf, zpoints, totalPoints);
    LOG4CPLUS_TRACE(logger, "ZPoints (" << totalPoints << ")");
    for (int i = 0; i < totalPoints; i++) {
      LOG4CPLUS_TRACE(logger, "  " << Kernels::buToString(buf[i]));
    }
  }
}

void Octree2::getUnorderedBCellFacetPairs() {
  static log4cplus::Logger logger =
    log4cplus::Logger::getInstance("Octree2.getUnorderedBCellFacetPairs");

  check(Kernels::GetBCellLCP_p(linesBuffer, zpoints, BCells,
    unorderedLineIndices, lines.size(), resln.mbits));

  if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
    vector<cl_int> buf;
    int debugSize = lines.size();
    Kernels::DownloadInts(unorderedLineIndices, buf, debugSize);
    LOG4CPLUS_TRACE(logger, "UnorderedBCellFacetIds (" << debugSize << ")");
    for (int i = 0; i < debugSize; i++) {
      LOG4CPLUS_TRACE(logger, "  " << buf[i]);
    }
    vector<BCell> buf2;
    Kernels::DownloadBCells(buf2, BCells, debugSize);
    LOG4CPLUS_TRACE(logger, "UnorderedBCellFacets (" << debugSize << ")");
    for (int i = 0; i < debugSize; i++) {
      LOG4CPLUS_TRACE(logger, "  " << Kernels::buToString(buf2[i].lcp)
        << " (" << buf2[i].lcpLength << ")");
    }
  }
}

void Octree2::getVertexOctree() {
  static log4cplus::Logger logger =
    log4cplus::Logger::getInstance("Octree2.buildVertexOctree");

  check(Kernels::BuildOctree_p(
    zpoints, totalPoints, octreeSize, resln.bits, resln.mbits));

  LOG4CPLUS_DEBUG(logger, "octreeSize = " << octreeSize);
}

void Octree2::getConflictCells() {

  cl::Buffer octreeBuffer = CLFW::Buffers["octree"];

  OctreeData od;
  od.fmin = bb.minimum;
  od.size = octreeSize;
  od.qwidth = resln.width;
  od.maxDepth = resln.bits;
  od.fwidth = bb.maxwidth;

  cl::Buffer facetPairs;
  check(Kernels::LookUpOctnodeFromBCell_p(BCells, octreeBuffer, unorderedNodeIndices, lines.size()));
  check(Kernels::RadixSortPairsByKey(unorderedNodeIndices, unorderedLineIndices, orderedNodeIndices, orderedLineIndices, lines.size()));
  check(Kernels::GetFacetPairs_p(orderedNodeIndices, facetPairs, lines.size(), octreeSize));
  //CLFW::DefaultQueue.finish();
  //static log4cplus::Logger logger =
  //  log4cplus::Logger::getInstance("Octree2.getConflictCells");
  //Timer t(logger, "GetConflictCells");
  check(Kernels::FindConflictCells_p(
    octreeBuffer, 
    leavesBuffer,
    totalLeaves,
    facetPairs, 
    od, 
    conflictsBuffer, 
    orderedLineIndices, 
    linesBuffer, 
    lines.size(), 
    quantizedPointsBuffer));
  //CLFW::DefaultQueue.finish();

}

void Octree2::getResolutionPoints() {
  if (lines.size() < 2) return;
  resolutionPoints.resize(0);
  resolutionPointsSize = 0;
  
  cl::Buffer conflictInfoBuffer, resolutionCounts,
    resolutionPredicates, scannedCounts;
  
  check(Kernels::GetResolutionPointsInfo_p(
    totalLeaves, conflictsBuffer, linesBuffer,
    quantizedPointsBuffer, conflictInfoBuffer, resolutionCounts,
    resolutionPredicates));
  check(CLFW::get(
    scannedCounts, "sResCnts",
    Kernels::nextPow2(totalLeaves) * sizeof(cl_int)));

  check(Kernels::StreamScan_p(
    resolutionCounts, scannedCounts, Kernels::nextPow2(totalLeaves),
    "resolutionIntermediate", false));

  check(CLFW::DefaultQueue.enqueueReadBuffer(
    scannedCounts, CL_TRUE,
    (totalLeaves * sizeof(cl_int)) - sizeof(cl_int), sizeof(cl_int),
    &resolutionPointsSize));
    
  if (resolutionPointsSize > 100000) {
    vector<cl_int> resolutionCountsVec;
    vector<cl_int> scannedCountsVec;
    DownloadInts(resolutionCounts, resolutionCountsVec, Kernels::nextPow2(totalLeaves));
    DownloadInts(scannedCounts, scannedCountsVec, Kernels::nextPow2(totalLeaves));
    cout << "Something likely went wrong. " << resolutionPointsSize << " seems like a lot of resolution points.." << endl;
    resolutionPointsSize = 0;
    return;
  }

  if (resolutionPointsSize < 0) {
    return;
  }
  check(Kernels::GetResolutionPoints_p(
    totalLeaves, resolutionPointsSize, conflictsBuffer, linesBuffer,
    quantizedPointsBuffer, conflictInfoBuffer, scannedCounts,
    resolutionPredicates, resolutionPointsBuffer));
}

void Octree2::addResolutionPoints() {
  benchmark("addResolutionPoints");
  if (resolutionPointsSize < 0) {
    cout << "Error computing resolution points." << endl;
    return;
  }
  if (resolutionPointsSize != 0) {
    int original = totalPoints;
    int additional = resolutionPointsSize;
    totalPoints += additional;
    cl_int error = 0;

    //If the new resolution points wont fit inside the existing buffer.
    if (original + additional > nextPow2(original)) {
      cl::Buffer oldQPointsBuffer = quantizedPointsBuffer;
      CLFW::Buffers["qPoints"] = cl::Buffer(CLFW::Contexts[0], CL_MEM_READ_WRITE, nextPow2(original + additional) * sizeof(intn));
      quantizedPointsBuffer = CLFW::Buffers["qPoints"];
      error |= CLFW::DefaultQueue.enqueueCopyBuffer(oldQPointsBuffer, quantizedPointsBuffer, 0, 0, original * sizeof(intn));
    }
    error |= CLFW::DefaultQueue.enqueueCopyBuffer(resolutionPointsBuffer, quantizedPointsBuffer, 0, original * sizeof(intn), additional * sizeof(intn));
    assert_cl_error(error);
  }
  if (Options::benchmarking) CLFW::DefaultQueue.finish();
}

/* Drawing Methods */
void Octree2::addOctreeNodes() {
  floatn temp;
  floatn center;

  if (octreeSize == 0) return;
  center = (bb.minimum + bb.maxwidth*.5);
  float3 color = { 0.75, 0.75, 0.75 };
  addOctreeNodes(0, center, bb.maxwidth, color);
}

void Octree2::addOctreeNodes(int index, floatn offset, float scale, float3 color)
{
  static log4cplus::Logger logger =
    log4cplus::Logger::getInstance("Octree2.addOctreeNodes");

  LOG4CPLUS_TRACE(logger, "offset = " << offset << " scale = " << scale
    << " index = " << index);

  Instance i = { offset.x, offset.y, 0.0, scale, color.x, color.y, color.z };
  gl_instances.push_back(i);
  if (index != -1) {
    OctNode current = octree[index];
    scale /= 2.0;
    float shift = scale / 2.0;
    addOctreeNodes(current.children[0], { offset.x - shift, offset.y - shift },
      scale, color);
    addOctreeNodes(current.children[1], { offset.x + shift, offset.y - shift },
      scale, color);
    addOctreeNodes(current.children[2], { offset.x - shift, offset.y + shift },
      scale, color);
    addOctreeNodes(current.children[3], { offset.x + shift, offset.y + shift },
      scale, color);
  }
}

void Octree2::addLeaf(int internalIndex, int childIndex, float3 color) {
  floatn center = BB_center(&bb);
  float octreeWidth = bb.maxwidth;
  OctNode node = octree[internalIndex];

  //Shift to account for the leaf
  float width = octreeWidth / (1 << (node.level + 1));
  float shift = width / 2.0;
  center.x += (childIndex & 1) ? shift : -shift;
  center.y += (childIndex & 2) ? shift : -shift;

  //Shift for the internal node
  while (node.parent != -1) {
    for (childIndex = 0; childIndex < 4; ++childIndex) {
      if (octree[node.parent].children[childIndex] == internalIndex)
        break;
    }

    shift *= 2.0;
    center.x += (childIndex & 1) ? shift : -shift;
    center.y += (childIndex & 2) ? shift : -shift;

    internalIndex = node.parent;
    node = octree[node.parent];
  }

  Instance i = {
    { center.x, center.y, 0.0 }
    , width
    , { color.x, color.y, color.z }
  };
  gl_instances.push_back(i);
}

void Octree2::addConflictCells() {
  // if (Options::debug) {
  Kernels::DownloadConflicts(conflicts, conflictsBuffer, totalLeaves);
  Kernels::DownloadLeaves(leavesBuffer, leaves, totalLeaves);
  if (conflicts.size() == 0) return;
  for (int i = 0; i < conflicts.size(); ++i) {
    if (conflicts[i].color == -2)
    {
      addLeaf(leaves[i].parent, leaves[i].zIndex,
      { Options::conflict_color[0],
        Options::conflict_color[1],
        Options::conflict_color[2] });
    }
  }
}

void Octree2::drawResolutionPoints() {
  using namespace GLUtilities;
  vector<intn> resolutionPoints;
  Kernels::DownloadQPoints(resolutionPoints, resolutionPointsBuffer, resolutionPointsSize);
  for (int i = 0; i < resolutionPoints.size(); ++i) {
    floatn point = UnquantizePoint(&resolutionPoints[i], &bb.minimum, resln.width, bb.maxwidth);
    Point p = {
      {point.x, point.y, 0.0, 1.0},
      {0.0, 0.0, 1.0, 1.0}
    };
    Sketcher::instance()->add(p);

  }
}

void Octree2::draw(const glm::mat4& mvMatrix) {
  Shaders::boxProgram->use();
  print_gl_error();
  glBindVertexArray(boxProgram_vao);
  print_gl_error();
  glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
  print_gl_error();
  glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * gl_instances.size(), gl_instances.data(), GL_STREAM_DRAW);
  print_gl_error();
  glUniformMatrix4fv(Shaders::boxProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
  glUniform1f(Shaders::boxProgram->pointSize_id, 10.0);
  print_gl_error();
  glLineWidth(2.0);
  ignore_gl_error();
  glDrawElementsInstanced(GL_LINES, 12 * 2, GL_UNSIGNED_BYTE, 0, gl_instances.size());
  print_gl_error();
  glBindVertexArray(0);
  print_gl_error();
}

#undef benchmark
#undef check