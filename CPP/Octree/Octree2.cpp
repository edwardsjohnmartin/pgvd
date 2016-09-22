#include <iostream>
#include <fstream>
#include <sstream>

#include <cstdio>

#ifdef __OPENCL_VERSION__
#define VEC_ACCESS(v, a) (v.a)
#define VEC_X(v) VEC_ACCESS(v, x)
#define VEC_Y(v) VEC_ACCESS(v, y)
#define VEC_Z(v) VEC_ACCESS(v, z)
#else
#define VEC_ACCESS(v, a) (v.s[a])
#define VEC_X(v) VEC_ACCESS(v, 0)
#define VEC_Y(v) VEC_ACCESS(v, 1)
#define VEC_Z(v) VEC_ACCESS(v, 2)
#endif

#include "./Octree2.h"
#include "../Karras/Karras.h" //TODO: parallelize quantization
//#include "../opencl/Geom.h"
//#include "../Kernels/Kernels.h"
//#include "../Timer/timer.h"
//
//extern "C" {
//#include "../../C/BinaryRadixTree/BuildBRT.h" //Does this need to be included?
//}
//
////OctCell fnode;
//
Octree2::Octree2() {
  const int n = 4;
  glm::vec3 drawVertices[n];
  drawVertices[0] = glm::vec3(-0.5, -0.5, 0);
  drawVertices[1] = glm::vec3( 0.5, -0.5, 0);
  drawVertices[2] = glm::vec3( 0.5,  0.5, 0);
  drawVertices[3] = glm::vec3(-0.5,  0.5, 0);

  glGenBuffers(1, &drawVertices_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, drawVertices_vbo);
  glBufferData(
      GL_ARRAY_BUFFER, n*sizeof(glm::vec3), drawVertices, GL_STATIC_DRAW);
	
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, drawVertices_vbo);

  glGenBuffers(1, &positions_vbo);
  glGenBuffers(1, &instance_vbo);
  glGenBuffers(1, &position_indices_vbo);
  glGenVertexArrays(1, &boxProgram_vao);
  glBindVertexArray(boxProgram_vao);
  glEnableVertexAttribArray(Shaders::boxProgram->position_id);
  glEnableVertexAttribArray(Shaders::boxProgram->offset_id);
  glEnableVertexAttribArray(Shaders::boxProgram->scale_id);
  glEnableVertexAttribArray(Shaders::boxProgram->color_id);
  assert (glGetError() == GL_NO_ERROR);
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
  assert(glGetError() == GL_NO_ERROR);
  glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
  assert(glGetError() == GL_NO_ERROR);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, position_indices_vbo);
  fprintf(stderr, "position_indices_vbo: %d\n", position_indices_vbo);
  assert(glGetError() == GL_NO_ERROR);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  assert(glGetError() == GL_NO_ERROR);
  glVertexAttribPointer(Shaders::boxProgram->position_id, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
  assert(glGetError() == GL_NO_ERROR);
  glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
  assert(glGetError() == GL_NO_ERROR);
  glVertexAttribPointer(Shaders::boxProgram->offset_id, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), 0);
  assert(glGetError() == GL_NO_ERROR);
  glVertexAttribPointer(Shaders::boxProgram->scale_id, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
  assert(glGetError() == GL_NO_ERROR);
  glVertexAttribPointer(Shaders::boxProgram->color_id, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(4 * sizeof(float)));
  assert(glGetError() == GL_NO_ERROR);
  glVertexAttribDivisor(Shaders::boxProgram->offset_id, 1);
  assert(glGetError() == GL_NO_ERROR);
  glVertexAttribDivisor(Shaders::boxProgram->scale_id, 1);
  assert(glGetError() == GL_NO_ERROR);
  glVertexAttribDivisor(Shaders::boxProgram->color_id, 1);
  glBindVertexArray(0);
  assert(glGetError() == GL_NO_ERROR);

  //resln = make_resln(1 << Options::max_level);
  resln = make_resln(1 << 6);
}

//int Octree2::processArgs(int argc, char** argv) {
//  
//  return 0;
//}

//void Octree2::build(const vector<float2>& points) {
//  using namespace std;
//  if (points.size() == 0) return;
//
//  karras_points.clear();
//  extra_qpoints.clear();
//  octree.clear();
//
//  karras_points = points;
//
//  // Compute bounding box
//  if (customBB) {
//    bb = *customBB;
//  }
//  else {
//    //Probably should be parallelized...
//    floatn minimum;
//    floatn maximum;
//    copy_fvfv(&minimum, &karras_points[0]);
//    copy_fvfv(&maximum, &karras_points[0]);
//    for (int i = 1; i < karras_points.size(); ++i) {
//      min_fvfv(&minimum, &minimum, &karras_points[i]);
//      max_fvfv(&maximum, &maximum, &karras_points[i]);
//    }
//    BB_initialize(&bb, &minimum, &maximum);
//  }
//
//  vector<intn> qpoints = Karras::Quantize(karras_points, resln, &bb);
//  if (qpoints.size() > 1) {
//    octree = Karras::BuildOctreeInParallel(qpoints, resln, false);
//  } else {
//    octree.clear();
//  }
//
//  //for (int i = 0; i < qpoints.size(); ++i) {
//  //  cout << qpoints[i] << endl;
//  //}
//
//  // Set up vertices on GPU for rendering
// // buildOctVertices();
//}
//
void Octree2::build(const PolyLines *polyLines) {
  using namespace std;

  karras_points.clear();
  extra_qpoints.clear();
  octree.clear();
  instances.clear();

  const vector<vector<float2>>& polygons = polyLines->getPolygons();
  lines = polyLines->getLines();

  if (polygons.empty()) return;
  
  // Get all vertices into a 1D array (karras_points).
  for (int i = 0; i < polygons.size(); ++i) {
    const vector<float2>& polygon = polygons[i];
    for (int j = 0; j < polygon.size()-1; ++j) {
      karras_points.push_back(polygon[j]);
    }
    karras_points.push_back(polygon.back());
  }

  for (int i = 0; i < lines.size(); ++i) {
    if(lines[i].secondIndex >= karras_points.size())
      lines = polyLines->getLines();
  }
  // Compute bounding box
  //Probably should be parallelized...
  floatn minimum;
  floatn maximum;
  copy_fvfv(&minimum, &karras_points[0]);
  copy_fvfv(&maximum, &karras_points[0]);
  for (int i = 1; i < karras_points.size(); ++i) {
    min_fvfv(&minimum, &minimum, &karras_points[i]);
    max_fvfv(&maximum, &maximum, &karras_points[i]);
  }
  BB_initialize(&bb, &minimum, &maximum);
  BB_make_centered_square(&bb, &bb);

  /* Draw Bounding Box for testing*/
  Instance i;
  floatn center;
  BB_center(&bb, &center);
  i.offset[0] = center.s[0];
  i.offset[1] = center.s[1];
  i.offset[2] = 0.0;
  float width;
  BB_max_size(&bb, &width);
  i.scale = width;
  i.color[0] = 1.0;
  i.color[1] = 1.0;
  i.color[2] = 1.0;
  instances.push_back(i);
  /* End of bounding box draw code*/

  // Karras iterations
  qpoints.clear();
  qpoints = Karras::Quantize((floatn*)karras_points.data(), karras_points.size(), resln, &bb);

//  int iterations = 0;
//  //do {
//    qpoints.insert(qpoints.end(), extra_qpoints.begin(), extra_qpoints.end());
    //for (const intn& qp : extra_qpoints) {
      //karras_points.push_back(oct2Obj(qp));
    //}
//    extra_qpoints.clear();
    if (qpoints.size() > 1) {
      octree = Karras::BuildOctreeInParallel(qpoints, resln, true);
    }
    else {
      octree.clear();
    }
//    //FindMultiCells(lines);
//
//   // ++iterations;
// // } while (iterations < Options::karras_iterations && !extra_qpoints.empty());
//  //cout << "Karras iterations: " << iterations << endl;
//
//  // Count the number of cells with multiple intersections
//  //int count = 0;
//  //for (int i = 0; i < cell_intersections.size(); ++i) {
//  //  for (int j = 0; j < 4; ++j) {
//  //    if (cell_intersections[i].is_multi(j)) {
//  //      ++count;
//  //    }
//  //  }
//  //}
//  //cout << "Number of multi-intersection cells: " << count << endl;
//  
  addOctreeNodes();
  if (lines.size() > 4) {
    findAmbiguousCells();
  }
}
//
//float2 Octree2::obj2Oct(const float2& v) const {
//  GLfloat bbw;
//  BB_max_size(&bb, &bbw);
//  float2 oct;
//  subt_fvfv(&oct, &v, &bb.minimum);
//  div_fvf(&oct, &oct, bbw);
//  mult_fvf(&oct, &oct, (float)resln.width);
//  //= ((v - bb.min()) / bbw) * (float)resln.width;
//  for (int i = 0; i < DIM; ++i) {
//    if (oct.s[i] >= resln.width)
//      oct.s[i] = resln.width - 0.0001;
//    if (oct.s[i] < 0)
//      throw logic_error("obj2Oct cannot have coord less than zero");
//  }
//  return oct;
//}
//
//glm::vec3 Octree2::toVec3(float2 p) const {
//  return glm::vec3(p.x, p.y, 0.0);
//}
//
//floatn Octree2::oct2Obj(const int2& v) const {
//  floatn vf = {v.s[0], v.s[1]};// make_float2(v.s[0], v.s[1]);
//  GLfloat bbw;
//  BB_max_size(&bb, &bbw);
//  // return (vf/kWidth)*bbw+bb.min();
//  mult_fvf(&vf, &vf, resln.width*bbw);
//  add_fvfv(&vf, &vf, &bb.minimum);
//  return vf;
//}
//
//GLfloat Octree2::oct2Obj(int dist) const {
//  GLfloat bbw;
//  BB_max_size(&bb, &bbw);
//  // const GLfloat ow = kWidth;
//  const GLfloat ow = resln.width;
//  return (dist/ow)*bbw;
//}
//
//// Point should be in object coordinates
//// void Octree2::Find(int x, int y) {
//void Octree2::Find(const float2& p) {
//  using namespace Karras;
//
//  // // floatn fv = Obj2Oct(Win2Obj(make_floatn(x, y)));
//  // floatn fv = obj2Oct(p);
//  // intn v = make_intn(fv.s[0], fv.s[1]);
//  // fnode = OctreeUtils::FindLeaf(v, octree, resln);
//  // dirty = true;
//  // glutPostRedisplay();
//}
//
////// Cell walk visitor
////// typedef void (*cwv)(Karras::OctCell, const intn& a, const intn& b,
////typedef void (*cwv)(OctCell, const floatn& a, const floatn& b,
////    const vector<OctNode>& octree, const Resln& resln, void* data);
//
//// Given a segment a-b, visit each octree cell that it intersects.
////void CellWalk(
////    const floatn& a, const floatn& b,
////    const vector<OctNode>& octree, const Resln& resln,
////    cwv v, void* data) {
////  using namespace Karras;
////
////  int dir[DIM];
////  for (int i = 0; i < DIM; ++i) {
////    if (b.s[i] > a.s[i]) dir[i] = 1;
////    else if (b.s[i] < a.s[i]) dir[i] = -1;
////    else dir[i] = 0;
////  }
////  OctreeUtils::CellIntersection cur(0, a);
////  const OctCell bcell = OctreeUtils::FindLeaf(convert_intn(b),
////                                                 octree, resln);
////  vector<OctreeUtils::CellIntersection> local_intersections;
////  vector<OctreeUtils::CellIntersection> all_intersections;
////  bool done = false;
////  int count = 0;
////  OctCell cell =
////      OctreeUtils::FindLeaf(convert_intn(cur.p), octree, resln);
////  do {
////    ++count;
////    if (count > 10000)
////      throw logic_error("Infinite loop");
////    v(cell, a, b, octree, resln, data);
////
////    const vector<OctreeUtils::CellIntersection> local_intersections =
////        OctreeUtils::FindIntersections(a, b, cell, /*octree,*/ resln);
////    for (const OctreeUtils::CellIntersection& i : local_intersections) {
////      if (i.t > cur.t) {
////        cur = i;
////      }
////    }
////    for (int i = 0; i < DIM; ++i) {
////      const int end = cell.get_origin().s[i];
////      if (cur.p.s[i] == end && cur.p.s[i] != 0 && dir[i] == -1) {
////        --cur.p.s[i];
////      }
////    }
////
////    OctCell new_cell = OctreeUtils::FindLeaf(
////        convert_intn(cur.p), octree, resln);
////    done = local_intersections.empty() ||
////        (cell.get_origin() == bcell.get_origin()) ||
////        (new_cell.get_origin() == cell.get_origin());
////    cell = new_cell;
////  } while (!done);
////}
//
////struct MCData {
////  MCData(vector<CellIntersections>& labels_) : labels(labels_) {}
////  vector<CellIntersections>& labels;
////  int cur_label;
////};
//
////void write_seg(const floatn& a, const floatn& b, const string& fn) {
////  using namespace std;
////  ofstream out(fn);
////  out << std::fixed << a << endl << b << endl;
////  out.close();
////}
//
////void write_seg(const FloatSegment& s, const string& fn) {
////  write_seg(s.a(), s.b(), fn);
////}
//
////void write_cell(const intn& o, const int& w, const string& fn) {
////  ofstream out(fn);
////  out << o << endl;
////  out << o+make_intn(w, 0) << endl;
////  out << o+make_intn(w, w) << endl;
////  out << o+make_intn(0, w) << endl;
////  out << o << endl;
////  out.close();
////}
//#if 0
////void error_log(OctCell cell, const floatn& a, const floatn& b,
////               const vector<OctNode>& octree, const Resln& resln) {
////  cerr << "a = " << std::fixed << a << endl;
////  cerr << "b = " << std::fixed << b << endl;
////  cerr << "cell = " << cell << endl;
////  //cerr << "resln = " << resln << endl;
////  ofstream out("mccallback.err");
////  //out << cell << " " << a << " " << b << " " << octree << " " << resln << endl;
////  out.close();
////  write_seg(a, b, "data1.dat");
////  const intn o = cell.get_origin();
////  const int w = cell.get_width();
////  write_cell(o, w, "data2.dat");
////  const int w2 = w/2;
////  // Boundary
////  write_cell(o-make_intn(w2, w2), 2*w, "data3.dat");
////}
//#endif
//inline bool within(const float& a, const float& b, const float& epsilon) {
//  return fabs(a-b) < epsilon;
//}
//#if 0
////// Multi-cell callback. Finds intersections with the current cell and
////// stores results in data.
////void MCCallback(OctCell cell, const floatn& a, const floatn& b,
////                  const vector<OctNode>& octree, const Resln& resln,
////                  void* data) {
////  // const static float EPSILON = 1e-6;
////
////  using namespace Karras;
////  MCData* d = static_cast<MCData*>(data);
////  const int octant = cell.get_octant();
////
////  // // debug
////  // OctCell test_cell;
////  // stringstream ss("8796416 6561024 256 22009 3 -1");
////  // ss >> test_cell;
////  // if (cell == test_cell || cell == fnode) {
////  //   cout << "here in MCCallback" << endl;
////  // }
////
////  // Get the intersections between the segment and octree cell.
////  const vector<OctreeUtils::CellIntersection> intersections =
////      OctreeUtils::FindIntersections(a, b, cell, /*octree,*/ resln);
////  if (intersections.size() > 2) {
////    error_log(cell, a, b, octree, resln);
////    cerr << "More than 2 intersections with a cell" << endl;
////    for (const OctreeUtils::CellIntersection& ci : intersections) {
////      cerr << "  " << ci.p << endl;
////    }
////    throw logic_error("More than 2 intersections with a cell");
////  }
////  if (intersections.size() == 2) {
////    FloatSegment seg(intersections[0].p,
////                  intersections[1].p);
////    CellIntersections& cell_intersections = d->labels[cell.get_parent_idx()];
////    cell_intersections.set(octant, d->cur_label, seg);
////  } else if (intersections.size() == 1) {
////    // Find which endpoint is in cell
////    const BoundingBox<intn> bb = cell.bb();
////    const floatn p = intersections[0].p;
////    floatn q;
////    bool degenerate = false;
////    // if (bb.in_half_open(convert_intn(a), EPSILON)) {
////    if (bb.in_half_open(convert_intn(a))) {
////      q = a;
////    // } else if (bb.in_half_open(convert_intn(b), EPSILON)) {
////    } else if (bb.in_half_open(convert_intn(b))) {
////      q = b;
////    } else if ((within(p.x, bb.min().x, EPSILON) ||
////                within(p.x, bb.max().x, EPSILON)) &&
////               (within(p.y, bb.min().y, EPSILON) ||
////                within(p.y, bb.max().y, EPSILON))) {
////      degenerate = true;
////    } else {
////      degenerate = true;
////      // error_log(cell, a, b, octree, resln);
////      // throw logic_error("Only one intersection but neither endpoint is"
////      //                   " in bounding box");
////    }
////    if (!degenerate) {
////      FloatSegment seg(p, q);
////      CellIntersections& cell_intersections = d->labels[cell.get_parent_idx()];
////      cell_intersections.set(octant, d->cur_label, seg);
////    }
////  }
////}
////
////void Octree2::FindMultiCells(const PolyLines& lines) {
////  using namespace Karras;
////
////  cell_intersections.clear();
////  cell_intersections.resize(octree.size(), CellIntersections());
////  MCData data(cell_intersections);
////
////  const vector<vector<float2>>& polygons = lines.getPolygons();;
////
////  // For each line segment in each polygon do a cell walk, updating
////  // the cell_intersections structure.
////  intersections.clear();
////  for (int j = 0; j < polygons.size(); ++j) {
////    const std::vector<float2>& polygon = polygons[j];
////    data.cur_label = j;
////    for (int i = 0; i < polygon.size() - 1; ++i) {
////      const floatn a = obj2Oct(polygon[i]);
////      const floatn b = obj2Oct(polygon[i+1]);
////      // Visit each octree cell intersected by segment a-b. The visitor
////      // is MCCallback.
////      CellWalk(a, b, octree, resln, MCCallback, &data);
////      vector<OctreeUtils::CellIntersection> local = Walk(a, b);
////      for (const OctreeUtils::CellIntersection& ci : local) {
////        intersections.push_back(ci.p);
////      }
////    }
////  }
////
////  // Find points that we want to add in order to run a more effective
////  // Karras octree construction.
////  extra_qpoints.clear();
////  _origins.clear();
////  _lengths.clear();
////  for (int i = 0; i < octree.size(); ++i) {
////    const CellIntersections& intersection = cell_intersections[i];
////    for (int octant = 0; octant < (1<<DIM); ++octant) {
////      const int num_labels = intersection.num_labels(octant);
////      // cout << "cell = " << i << "/" << octant << ": " << num_labels << endl;
////      // if (intersection.is_multi(octant)) {
////      //   {
////      for (int j = 0; j < num_labels; ++j) {
////      // for (int j = 0; j < 1; ++j) {
////        for (int k = j+1; k < num_labels; ++k) {
////        // for (int k = j+1; k < std::min(2, num_labels); ++k) {
////          // We'll compare segments at j and k.
////          FloatSegment segs[2] = { intersection.seg(j, octant),
////                                intersection.seg(k, octant) };
////          vector<floatn> samples;
////          vector<floatn> origins;
////          vector<float> lengths;
////          if (!Geom::multi_intersection(segs[0], segs[1])) {
////            try {
////              if (i == fnode.get_parent_idx() && octant == fnode.get_octant()) {
////                cout << "here in FitBoxes" << endl;
////                write_seg(segs[0], "seg0.dat");
////                write_seg(segs[1], "seg1.dat");
////              }
////              Geom::FitBoxes(segs[0], segs[1], 1, &samples, &origins, &lengths);
////            } catch(logic_error& e) {
////              cerr << "segments: " << segs[0] << " " << segs[1] << endl;
////              cerr << "labels: " << intersection.label(0, octant)
////                   << " " << intersection.label(1, octant) << endl;
////              write_seg(segs[0], "seg0.dat");
////              write_seg(segs[1], "seg1.dat");
////              throw e;
////            }
////          }
////          _origins.insert(_origins.end(), origins.begin(), origins.end());
////          _lengths.insert(_lengths.end(), lengths.begin(), lengths.end());
////          for (const floatn& sample : samples) {
////            extra_qpoints.push_back(convert_intn(sample));
////          }
////        }
////      }
////    }
////  }
////}
//
////// void WalkCallback(Karras::OctCell cell, const intn& a, const intn& b,
////void WalkCallback(OctCell cell, const floatn& a, const floatn& b,
////                  const vector<OctNode>& octree, const Resln& resln,
////                  void* data) {
////  
////  using namespace Karras;
////  using OctreeUtils::CellIntersection;
////  using OctreeUtils::FindIntersections;
////
////  vector<CellIntersection>& all_intersections =
////      *static_cast<vector<CellIntersection>*>(data);
////
////  const vector<CellIntersection> local_intersections =
////      FindIntersections(a, b, cell, /*octree,*/ resln);
////  for (const CellIntersection& i : local_intersections) {
////    // const intn p = i.p;
////    all_intersections.push_back(i);
////  }
////}
////
////vector<OctreeUtils::CellIntersection> Octree2::Walk(
////    // const intn& a, const intn& b) {
////    const floatn& a, const floatn& b) {
////  using namespace Karras;
////
////  vector<OctreeUtils::CellIntersection> all_intersections;
////  CellWalk(a, b, octree, resln, WalkCallback, &all_intersections);
////
////  return all_intersections;
////}
//#endif
////Used purely for testing. 
//void Octree2::getZPoints(vector<BigUnsigned> &zpoints, const std::vector<intn> &qpoints) {
//  zpoints.clear();
//  zpoints.resize(qpoints.size());
//  cl::Buffer zpointsBuffer = CLFW::Buffers["zpoints"];
//  cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(zpointsBuffer, CL_TRUE, 0, qpoints.size()*sizeof(BigUnsigned), zpoints.data());
//}
//
/* Drawing Methods */
void Octree2::addOctreeNodes() {
  float2 temp;
  floatn center;
  float width;

  if (octree.size() == 0) return;
  BB_max_size(&bb, &width);
  copy_fvf(&temp, width / 2.0F);
  add_fvfv(&center, &bb.minimum, &temp);
  cl_float3 color = { 1.0, 1.0, 1.0 };
  cout << endl;
  addOctreeNodes(0, center, width, color);
}
void Octree2::addOctreeNodes(int index, floatn offset, float scale, float3 color) {
  Instance i = { offset.x, offset.y, 0.0, scale, VEC_X(color), VEC_Y(color), VEC_Z(color) };
  instances.push_back(i);
  if (index != -1) {
    OctNode current = octree[index];
    scale /= 2.0;
    float shift = scale / 2.0;
    addOctreeNodes(current.children[0], { offset.x - shift, offset.y - shift }, scale, color);
    addOctreeNodes(current.children[1], { offset.x + shift, offset.y - shift }, scale, color);
    addOctreeNodes(current.children[2], { offset.x - shift, offset.y + shift }, scale, color);
    addOctreeNodes(current.children[3], { offset.x + shift, offset.y + shift }, scale, color);
  }
}

void Octree2::addLeaf(int internalIndex, int childIndex, float3 color) {
  floatn center;
  BB_center(&bb, &center);
  float octreeWidth;
  BB_max_size(&bb, &octreeWidth);
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
    , { VEC_X(color), VEC_Y(color), VEC_Z(color) }
  };
  instances.push_back(i);
}

void Octree2::findAmbiguousCells() {
  if (qpoints.size() < 2) return;
  cl::Buffer linesBuffer, sortedLines, boundingBoxesBuffer;
  cl::Buffer octreeBuffer = CLFW::Buffers["octree"];
  cl::Buffer zpoints = CLFW::Buffers["zpoints"];
  vector<int> boundingBoxes;
  vector<Line> sortedLines_vec(lines.size());
  assert(Kernels::UploadLines(lines, linesBuffer) == CL_SUCCESS);
  vector<BigUnsigned> zpoints_vec(qpoints.size());
  CLFW::DefaultQueue.enqueueReadBuffer(zpoints, CL_TRUE, 0, qpoints.size() * sizeof(BigUnsigned), zpoints_vec.data());\
  assert(Kernels::ComputeLineLCPs_p(linesBuffer, zpoints, lines.size(), resln.mbits) == CL_SUCCESS);
  assert(Kernels::RadixSortLines_p(linesBuffer, sortedLines, lines.size(), resln.mbits) == CL_SUCCESS);
  assert(Kernels::ComputeLineBoundingBoxes_p(sortedLines, octreeBuffer, boundingBoxesBuffer, lines.size()) == CL_SUCCESS);
  assert(Kernels::DownloadLines(sortedLines, sortedLines_vec, sortedLines_vec.size()) == CL_SUCCESS);
  assert(Kernels::DownloadBoundingBoxes(boundingBoxesBuffer, boundingBoxes, lines.size()) == CL_SUCCESS);

  vector<int> leafColors(4 * octree.size(), -1);
  floatn octreeCenter = {};
  float width;
  BB_center(&bb, &octreeCenter);
  BB_max_size(&bb, &width);
  for (unsigned int i = 0; i < octree.size(); ++i) {
    Kernels::FindAmbiguousCells_p(octree.data(), octree.size(), octreeCenter, width, leafColors.data(), 
      boundingBoxes.data(), boundingBoxes.size(), sortedLines_vec.data(), sortedLines_vec.size(), karras_points.data(), i);
  }

  for (int i = 0; i < octree.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      if (leafColors[i * 4 + j] == -2) {
        addLeaf(i, j, { 1.0, 0.0, 0.0 });
      }
    }
  }
}
//void Octree2::addNode(BigUnsigned lcp, int lcpLength, float colorStrength) {
//  using namespace std;
//  //Some special cases.
//  if (octree.size() == 0) return;
//  if (lcpLength == resln.mbits) return;
//
//  //Helpful information
//  float width;
//  BB_max_size(&bb, &width);
//  floatn center;
//  float2 temp;
//  copy_fvf(&temp, width / 2.0F);
//  add_fvfv(&center, &bb.minimum, &temp);
//  int numLevels = lcpLength / DIM;
//  int isOdd = (lcpLength & 1 == 1) ? 1 : 0;
//  int indx = 0;
//
//  //calculate width and offset by itterating through the tree
//  for (int i = 0; i < numLevels; ++i) {
//    //Get index
//    if (lcp.len != 0) {
//      BigUnsigned mask;
//      BigUnsigned result;
//      int shiftAmount = (numLevels - i - 1) * DIM + isOdd;
//      initBlkBU(&mask, ((DIM == 2) ? 3 : 7));
//      shiftBULeft(&mask, &mask, shiftAmount);
//      andBU(&result, &mask, &lcp);
//      shiftBURight(&result, &result, shiftAmount);
//      indx = result.blk[0];
//
//      width /= 2.0;
//      switch (indx) {
//      case 0:
//        add_fvfv_by_val(&center, center, { -width / 2.0F, -width / 2.0F });
//        break;
//      case 1:
//        add_fvfv_by_val(&center, center, { width / 2.0F,-width / 2.0F });
//        break;
//      case 2:
//        add_fvfv_by_val(&center, center, { -width / 2.0F,width / 2.0F });
//        break;
//      case 3:
//        add_fvfv_by_val(&center, center, { width / 2.0F,width / 2.0F });
//        break;
//      }
//    }
//  }
//  
//  instances.push_back({
//    center.x, center.y, 0.0,
//    width,
//    colorStrength, 0.0, 0.0,
//  });
//}
void Octree2::draw() {
  Shaders::boxProgram->use();
  glBindVertexArray(boxProgram_vao);
  glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * instances.size(), instances.data(), GL_STREAM_DRAW);
  glm::mat4 identity(1.0);
  glUniformMatrix4fv(Shaders::boxProgram->matrix_id, 1, 0, &(identity[0].x)); //glm::value_ptr wont work on identity for some reason...
  glUniform1f(Shaders::boxProgram->pointSize_id, 10.0);
  assert(glGetError() == GL_NO_ERROR);
  glLineWidth( 2.0);
  glDrawElementsInstanced(GL_LINES, 12 * 2, GL_UNSIGNED_BYTE, 0, instances.size());
  glBindVertexArray(0);
}
