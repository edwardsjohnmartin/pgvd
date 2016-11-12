#include <iostream>
#include <fstream>
#include <sstream>

#include <cstdio>

#include "../GLUtilities/gl_utils.h"
#include "./Octree2.h"
#include "../Timer/timer.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

Octree2::Octree2() {
    const int n = 4;
    /*glm::vec3 drawVertices[n];
    drawVertices[0] = glm::vec3(-0.5, -0.5, 0);
    drawVertices[1] = glm::vec3(0.5, -0.5, 0);
    drawVertices[2] = glm::vec3(0.5, 0.5, 0);
    drawVertices[3] = glm::vec3(-0.5, 0.5, 0);*/

    /*glGenBuffers(1, &drawVertices_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, drawVertices_vbo);
    glBufferData( GL_ARRAY_BUFFER, n * sizeof(glm::vec3), drawVertices, GL_STATIC_DRAW);*/

    /*glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, drawVertices_vbo);*/

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

    // resln = make_resln(1 << 8);
    resln = make_resln(1 << Options::max_level);
}

inline floatn min_floatn(const floatn a, const floatn b) {
    floatn result;
    for (int i = 0; i < DIM; ++i) {
        result.s[i] = (a.s[i] < b.s[i]) ? a.s[i] : b.s[i];
    }
    return result;
}

inline floatn max_floatn(const floatn a, const floatn b) {
    floatn result;
    for (int i = 0; i < DIM; ++i) {
        result.s[i] = (a.s[i] > b.s[i]) ? a.s[i] : b.s[i];
    }
    return result;
}

void Octree2::generatePoints(const PolyLines *polyLines) {
    const vector<vector<floatn>> polygons = polyLines->getPolygons();
    lines = polyLines->getLines();

    // Get all vertices into a 1D array (karras_points).
    for (int i = 0; i < polygons.size(); ++i) {
        const vector<floatn>& polygon = polygons[i];
        for (int j = 0; j < polygon.size(); ++j) {
            karras_points.push_back(polygon[j]);
        }
    }
    /*cout << "num points = " << karras_points.size() << endl;
    for (int i = 0; i < karras_points.size(); ++i) {
      cout << karras_points[i] << endl;
    }*/
}

void Octree2::computeBoundingBox(const int totalPoints) {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Octree2.computeBoundingBox");

  if (Options::xmin == -1 && Options::xmax == -1) {
    //Probably should be parallelized...
    floatn minimum = karras_points[0];
    floatn maximum = karras_points[0];
    for (int i = 1; i < totalPoints; ++i) {
      minimum = min_floatn(karras_points[i], minimum);
      maximum = max_floatn(karras_points[i], maximum);
    }
    bb = BB_initialize(&minimum, &maximum);
    bb = BB_make_centered_square(&bb);
  } else {
    bb.initialized = true;
    bb.minimum = make_floatn(Options::xmin, Options::ymin);
    bb.maximum = make_floatn(Options::xmax, Options::ymax);
    bb.maxwidth = BB_max_size(&bb);
  }
  LOG4CPLUS_DEBUG(logger, bb);
}

void Octree2::quantizePoints(int numResolutionPoints) {
    Kernels::QuantizePoints_p(totalPoints, karrasPointsBuffer, quantizedPointsBuffer, bb.minimum, resln.width, bb.maxwidth);
}

void Octree2::makeZOrderPoints() {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Octree2.makeZOrderPoints");
  //Timer t("Making Z Order points.");

  Kernels::PointsToMorton_p(
      quantizedPointsBuffer, zpoints, totalPoints, resln.bits);

  //CLFW::DefaultQueue.finish();
  if (logger.isEnabledFor(log4cplus::TRACE_LOG_LEVEL)) {
    vector<BigUnsigned> buf;
    Kernels::DownloadZPoints(buf, zpoints, totalPoints);
    LOG4CPLUS_TRACE(logger, "ZPoints (" << totalPoints << ")");
    for (int i = 0; i < totalPoints; i++) {
      LOG4CPLUS_TRACE(logger, "  " << Kernels::buToString(buf[i]));
    }
  }
}

void Octree2::GetUnorderedBCellFacetPairs() {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Octree2.getUnorderedBCellFacetPairs");

  Kernels::GetBCellLCP_p(linesBuffer, zpoints, BCells,
                         unorderedLineIndices, lines.size(), resln.mbits);
    
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

void Octree2::buildVertexOctree() {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Octree2.buildVertexOctree");

    Kernels::BuildOctree_p(
        zpoints, totalPoints, octreeSize, resln.bits, resln.mbits);

    LOG4CPLUS_DEBUG(logger, "octreeSize = " << octreeSize);
}

void Octree2::identifyConflictCells() {
    //Timer t("Identifying conflicts");
    cl::Buffer octreeBuffer = CLFW::Buffers["octree"];

    OctreeData od;
    od.fmin = bb.minimum;
    od.size = octreeSize;
    od.qwidth = resln.width;
    od.maxDepth = resln.bits;
    od.fwidth = bb.maxwidth;

    cl::Buffer facetPairs;
    cl_int error = Kernels::LookUpOctnodeFromBCell_p(BCells, octreeBuffer, unorderedNodeIndices, lines.size());
    error |= Kernels::RadixSortPairsByKey(unorderedNodeIndices, unorderedLineIndices, orderedNodeIndices, orderedLineIndices, lines.size());
    
    error |= Kernels::GetFacetPairs_p(orderedNodeIndices, facetPairs, lines.size(), octreeSize);
    error |= Kernels::FindConflictCells_p(octreeBuffer, facetPairs, od, conflictsBuffer, orderedLineIndices, linesBuffer, lines.size(), quantizedPointsBuffer);
    //assert(CLFW::Queues[0].finish() == CL_SUCCESS);
    
}

void Octree2::getResolutionPoints() {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Octree2.getResolutionPoints");

  LOG4CPLUS_TRACE(logger, "Calling");

  //Timer t("Getting resolution points");
  if (lines.size() < 2) return;
  resolutionPoints.resize(0);

  resolutionPointsSize = 0;
  //Parallel version
  cl::Buffer conflictInfoBuffer, resolutionCounts,
      resolutionPredicates, scannedCounts;
  bool success;
  success = Kernels::GetResolutionPointsInfo_p(
      octreeSize, conflictsBuffer, linesBuffer,
      quantizedPointsBuffer, conflictInfoBuffer, resolutionCounts,
      resolutionPredicates);
  assert_cl_error(success);
  CLFW::DefaultQueue.finish();

  vector<ConflictInfo> conflictInfoBuffer_s(octreeSize*4);
  vector<Conflict> conflicts_s;
  if (Options::series) {
    // Serial version
    unsigned int totalOctnodes_s = octreeSize;
    vector<Line> orderedLines_s;
    vector<intn> qPoints_s;
    vector<unsigned int> resolutionCounts_s(totalOctnodes_s*4);
    vector<int> predicates_s(totalOctnodes_s*4);
    Kernels::DownloadConflicts(conflicts_s, conflictsBuffer, totalOctnodes_s*4);
    Kernels::DownloadLines(linesBuffer, orderedLines_s, lines.size());
    Kernels::DownloadQPoints(
        qPoints_s, quantizedPointsBuffer, karras_points.size());

    success |= Kernels::GetResolutionPointsInfo_s(
        totalOctnodes_s, conflicts_s.data(), orderedLines_s.data(),
        qPoints_s.data(), conflictInfoBuffer_s.data(),
        resolutionCounts_s.data(), predicates_s.data());

    for (int i = 0; i < octreeSize*4; ++i) {
      if (conflictInfoBuffer_s[i].num_samples < 0) {
        LOG4CPLUS_WARN(logger, "conflictInfo_s " << conflictInfoBuffer_s[i]);
        exit(0);
      }
    }
  }


  if (logger.isEnabledFor(log4cplus::DEBUG_LOG_LEVEL)) {
    vector<ConflictInfo> buf;
    int bsize = octreeSize * 4;
    Kernels::DownloadConflictInfo(buf, conflictInfoBuffer, bsize);
    LOG4CPLUS_TRACE(logger, "ConflictInfo (" << bsize << ")");

    int count = 0;
    for (int i = 0; i < bsize; ++i) {
      if (buf[i].num_samples > 0) {
        // LOG4CPLUS_TRACE(logger, "  " << buf[i]);
      } else if (buf[i].num_samples < 0) {
        // LOG4CPLUS_ERROR(logger, "num_samples = " << buf[i].num_samples);
        LOG4CPLUS_WARN(logger, "conflictInfo_p " << buf[i]);
      }
      if (buf[i].num_samples == 0 && buf[i].padding == -2) {
      // if (buf[i].currentNode == 712) {
        LOG4CPLUS_WARN(logger, buf[i]);
        // if (buf[i].num_line_pairs > 0) {
        //   exit(0);
        // }
        if (buf[i].currentNode == 57) {
          exit(0);
        }
        // count++;
        // if (count > 5) {
        // }
      }
    }
  }

  LOG4CPLUS_TRACE(logger, "Retrieved ConflictInfo objects");

  vector<int> test(1);
  CLFW::DefaultQueue.enqueueReadBuffer(
      resolutionCounts, CL_TRUE, 0, sizeof(cl_int), test.data());
  // cout << test[0] << " =? " << sizeof(ConflictInfo) << endl;
  // cout << "Test = " << test[0] << endl;

  LOG4CPLUS_TRACE(logger, "Getting scannedCounts");
  success = CLFW::get(
      scannedCounts, "sResCnts",
      Kernels::nextPow2(octreeSize * 4) * sizeof(cl_int));
  assert_cl_error(success);

  LOG4CPLUS_TRACE(logger, "Doing stream scan");
  success = Kernels::StreamScan_p(
      resolutionCounts, scannedCounts, Kernels::nextPow2(4 * octreeSize),
      "resolutionIntermediate", false);
  assert_cl_error(success);

  LOG4CPLUS_TRACE(logger, "Reading resolutionPointsSize "
                  << octreeSize);
  success = CLFW::DefaultQueue.enqueueReadBuffer(
      scannedCounts, CL_TRUE,
      (octreeSize * 4 * sizeof(cl_int)) - sizeof(cl_int), sizeof(cl_int),
      &resolutionPointsSize);
  assert_cl_error(success);
  LOG4CPLUS_TRACE(logger, "Read scannedCounts " << resolutionPointsSize);

  if (resolutionPointsSize < 0) {
    return;
  }

  LOG4CPLUS_TRACE(logger, "Performed scan");

  success = Kernels::GetResolutionPoints_p(
      octreeSize, resolutionPointsSize, conflictsBuffer, linesBuffer,
      quantizedPointsBuffer, conflictInfoBuffer, scannedCounts,
      resolutionPredicates, resolutionPointsBuffer);
  assert_cl_error(success);

  vector<intn> gpuResolutionPoints(resolutionPointsSize);
  CLFW::DefaultQueue.enqueueReadBuffer(
      resolutionPointsBuffer, CL_TRUE, 0,
      sizeof(intn)*resolutionPointsSize, gpuResolutionPoints.data());

  LOG4CPLUS_TRACE(logger, "2");

  using namespace GLUtilities;
  for (int i = 0; i < gpuResolutionPoints.size(); ++i) {
    intn q = gpuResolutionPoints[i];
    floatn newp = UnquantizePoint(&q, &bb.minimum, resln.width, bb.maxwidth);

    resolutionPoints.push_back(q);

    GLUtilities::Point p =
        { { newp.x, newp.y , 0.0, 1.0 },{ 1.0,0.0,0.0,1.0 } };
    Sketcher::instance()->add(p);
  }
  LOG4CPLUS_TRACE(logger, "3");
  CLFW::DefaultQueue.finish();
}

void Octree2::insertResolutionPoints() {
  //Timer t("Inserting resolution points.");
  if (resolutionPointsSize < 0) {
    cout << "Error computing resolution points." << endl;
    return;
  }
  using namespace Kernels;
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

  //CLFW::DefaultQueue.finish();
}

void Octree2::build(const PolyLines *polyLines) {
  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("Octree2.build");

  Timer t(logger, "initialize");
  using namespace std;
  using namespace GLUtilities;
  using namespace Kernels;
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
  generatePoints(polyLines);
  if (karras_points.size() == 0) return;

  t.restart("quantize");
  totalPoints = karras_points.size();
  computeBoundingBox(totalPoints);
  UploadKarrasPoints(karras_points, karrasPointsBuffer);
  UploadLines(lines, linesBuffer);
  quantizePoints();

  LOG4CPLUS_TRACE(logger, "1");

  makeZOrderPoints();
  t.restart("resolve conflicts");
  do {
    Timer itTimer(logger, "Iteration");
    totalIterations++;
    LOG4CPLUS_INFO(logger, "Iteration " << totalIterations
                   << " (oct size: " << octreeSize << ")...");

    CLFW::DefaultQueue = CLFW::Queues[0];
    previousSize = octreeSize;

    LOG4CPLUS_TRACE(logger, "2");
    insertResolutionPoints();
    makeZOrderPoints(); //TODO: only z-order the additional points here...
    CLFW::DefaultQueue.finish();
    /*
      On one queue, sort the lines for the ambiguous cell detection.
      On the other, start building the karras octree.
    */
    LOG4CPLUS_TRACE(logger, "3");
    CLFW::DefaultQueue = CLFW::Queues[1];
    GetUnorderedBCellFacetPairs(); //13% (concurrent)
    CLFW::DefaultQueue = CLFW::Queues[0];
    buildVertexOctree(); //40% (concurrent)
    CLFW::Queues[0].finish();
    CLFW::Queues[1].finish();

    LOG4CPLUS_TRACE(logger, "4");
    /* Identify the cells that contain more than one object. */
    if (lines.size() == 0 || lines.size() == 1) break;
    identifyConflictCells(); //43%

    LOG4CPLUS_TRACE(logger, "5");
    /* Resolve one iteration of conflicts. */
    getResolutionPoints();//12%

    LOG4CPLUS_TRACE(logger, "6");

    if (totalIterations == Options::maxConflictIterations) {
      LOG4CPLUS_WARN(logger, "*** Warning: breaking out of conflict "
                     << "detection loop early for debugging purposes.");
      // exit(0);
      break;
    }
  } while (previousSize != octreeSize && resolutionPointsSize != 0);

  LOG4CPLUS_INFO(logger, "Total iterations: " << totalIterations);
  LOG4CPLUS_INFO(logger, "Octree size: " << octreeSize);

  t.restart("downloading and drawing octree");

  // LOG4CPLUS_WARN(logger, "*** Not drawing octree.");
  // return;

  cl::Buffer octreeBuffer = CLFW::Buffers["octree"];
  octree.resize(octreeSize);
  cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(
      octreeBuffer, CL_TRUE, 0, sizeof(OctNode)*octreeSize, octree.data());
  assert_cl_error(error);

  /* Add the octnodes so they'll be rendered with OpenGL. */
  addOctreeNodes();
  /* Add conflict cells so they'll be rendered with OpenGL. */
  addConflictCells();
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

void Octree2::addOctreeNodes(
    int index, floatn offset, float scale, float3 color) {
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
    Kernels::DownloadConflicts(conflicts, conflictsBuffer, octree.size() * 4);
    // cout << "Conflicts" << endl;
    // for (int i = 0; i < conflicts.size(); i++) {
    //   cout << "  " << conflicts[i] << endl;
    // }
  // }
  if (conflicts.size() == 0) return;
  for (int i = 0; i < octreeSize; ++i) {
    for (int j = 0; j < 4; j++) {
      if (conflicts[4 * i + j].color == -2) {
        // addLeaf(i, j, { 1.0, 0.0, 0.0 });
        addLeaf(i, j, { Options::conflict_color[0],
                Options::conflict_color[1],
                Options::conflict_color[2] });
      }
    }
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
