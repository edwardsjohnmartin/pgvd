#include <iostream>
#include <fstream>
#include <sstream>

#include "./Octree2.h"
#include "../Karras.h"
#include "../opencl/Geom.h"

OctCell fnode;

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

  // Set up indices
  numIndices = 0;
}

int Octree2::processArgs(int argc, char** argv) {
  int i = 1;
  bool stop = false;
  while (i < argc && !stop) {
    stop = true;
    if (options.ProcessArg(i, argv)) {
      stop = false;
    }
  }
  // if (options.help) {
  //   PrintUsage();
  //   exit(0);
  // }

  // resln = Resln(1<<options.max_level);
  resln = make_resln(1 << options.max_level);
  //resln = make_resln(1 << 2);

  // if (options.test > -1)
  //   test(options.test);

  for (; i < argc; ++i) {
    std::string filename(argv[i]);
    std::cout << filename << std::endl;
    options.filenames.push_back(filename);
  }

  // if (options.filenames.empty()) {
  //   throw logic_error("No filenames");
  // }

  // Moved to main_pgvd
  // for (const string& f : options.filenames) {
  //   // ReadMesh(f);
  //   ifstream in(f.c_str());
  //   bool first = true;
  //   while (!in.eof()) {
  //     double x, y;
  //     in >> x >> y;
  //     if (!in.eof()) {
  //       if (first) {
  //       }
  //     }
  //   }
  //   in.close();
  // }

  // int num_edges = 0;
  // for (int i = 0; i < polygons.size(); ++i) {
  //   num_edges += polygons[i].size();
  // }

  // PrintCommands();
  // cout << "Number of objects: " << polygons.size() << endl;
  // cout << "Number of polygon edges: " << num_edges << endl;

  return 0;
}

void Octree2::build(const vector<float2>& points,
                    const BoundingBox<float2>* customBB) {
  using namespace std;

  karras_points.clear();
  bb = BoundingBox<float2>();
  extra_qpoints.clear();
  octree.clear();

  karras_points = points;

  // Compute bounding box
  if (customBB) {
    bb = *customBB;
  } else {
    for (int i = 0; i < karras_points.size(); ++i) {
      bb(karras_points[i]);
    }
  }

  vector<intn> qpoints = Karras::Quantize(karras_points, resln, &bb);
  if (qpoints.size() > 1) {
    octree = Karras::BuildOctreeInParallel(qpoints, resln, false);
  } else {
    octree.clear();
  }

  for (int i = 0; i < qpoints.size(); ++i) {
    cout << qpoints[i] << endl;
  }

  // Set up vertices on GPU for rendering
  //buildOctVertices();
}

void Octree2::build(const Polylines& lines,
                    const BoundingBox<float2>* customBB) {
  using namespace std;

  //------------------
  // Initialize OpenCL
  //------------------
// #ifdef __OPEN_CL_SUPPORT__
//   static bool initialized = false;
//   if (options.gpu && !initialized) {
//     OpenCLInit(2, o, options.opencl_log);
//     initialized = true;
//   }
// #endif

  karras_points.clear();
  bb = BoundingBox<float2>();
  extra_qpoints.clear();
  octree.clear();

  const vector<vector<float2>>& polygons = lines.getPolygons();
  if (polygons.empty()) {
    buildOctVertices();
    return;
  }

  // Get all vertices into a 1D array (karras_points).
  for (int i = 0; i < polygons.size(); ++i) {
    const vector<float2>& polygon = polygons[i];
    for (int j = 0; j < polygon.size()-1; ++j) {
      karras_points.push_back(polygon[j]);
    }
    karras_points.push_back(polygon.back());
  }

  // Compute bounding box
  if (customBB) {
    bb = *customBB;
  } else {
    for (int i = 0; i < karras_points.size(); ++i) {
      bb(karras_points[i]);
    }
  }

  // Karras iterations
  vector<intn> qpoints = Karras::Quantize(karras_points, resln);
  int iterations = 0;
  do {
    qpoints.insert(qpoints.end(), extra_qpoints.begin(), extra_qpoints.end());
    for (const intn& qp : extra_qpoints) {
      karras_points.push_back(oct2Obj(qp));
    }
    extra_qpoints.clear();
    if (qpoints.size() > 1) {
      octree = Karras::BuildOctreeInParallel(qpoints, resln, false);
    } else {
      octree.clear();
    }
    //FindMultiCells(lines);

    ++iterations;
  } while (iterations < options.karras_iterations && !extra_qpoints.empty());
  //cout << "Karras iterations: " << iterations << endl;

  // Count the number of cells with multiple intersections
  int count = 0;
  for (int i = 0; i < cell_intersections.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      if (cell_intersections[i].is_multi(j)) {
        ++count;
      }
    }
  }
  //cout << "Number of multi-intersection cells: " << count << endl;

  // todo: setup vertices on GPU for rendering
  //buildOctVertices();
  
  //------------------
  // Cleanup OpenCL
  //------------------
// #ifdef __OPEN_CL_SUPPORT__
//   if (options.gpu) {
//     OpenCLCleanup();
//   }
// #endif
}

float2 Octree2::obj2Oct(const float2& v) const {
  const GLfloat bbw = bb.max_size();
  float2 oct = ((v-bb.min())/bbw) * (float)resln.width;
  for (int i = 0; i < DIM; ++i) {
    if (oct.s[i] >= resln.width)
      oct.s[i] = resln.width - 0.0001;
    if (oct.s[i] < 0)
      throw logic_error("obj2Oct cannot have coord less than zero");
  }
  return oct;
}

glm::vec3 Octree2::toVec3(float2 p) const {
  return glm::vec3(p.x, p.y, 0.0);
}

float2 Octree2::oct2Obj(const int2& v) const {
  const float2 vf = make_float2(v.s[0], v.s[1]);
  const GLfloat bbw = bb.max_size();
  // return (vf/kWidth)*bbw+bb.min();
  return (vf/resln.width)*bbw+bb.min();
}

GLfloat Octree2::oct2Obj(int dist) const {
  const GLfloat bbw = bb.max_size();
  // const GLfloat ow = kWidth;
  const GLfloat ow = resln.width;
  return (dist/ow)*bbw;
}

// Point should be in object coordinates
// void Octree2::Find(int x, int y) {
void Octree2::Find(const float2& p) {
  using namespace Karras;

  // // floatn fv = Obj2Oct(Win2Obj(make_floatn(x, y)));
  // floatn fv = obj2Oct(p);
  // intn v = make_intn(fv.s[0], fv.s[1]);
  // fnode = OctreeUtils::FindLeaf(v, octree, resln);
  // dirty = true;
  // glutPostRedisplay();
}

// Cell walk visitor
// typedef void (*cwv)(Karras::OctCell, const intn& a, const intn& b,
typedef void (*cwv)(OctCell, const floatn& a, const floatn& b,
    const vector<OctNode>& octree, const Resln& resln, void* data);

// Given a segment a-b, visit each octree cell that it intersects.
void CellWalk(
    const floatn& a, const floatn& b,
    const vector<OctNode>& octree, const Resln& resln,
    cwv v, void* data) {
  using namespace Karras;

  int dir[DIM];
  for (int i = 0; i < DIM; ++i) {
    if (b.s[i] > a.s[i]) dir[i] = 1;
    else if (b.s[i] < a.s[i]) dir[i] = -1;
    else dir[i] = 0;
  }
  OctreeUtils::CellIntersection cur(0, a);
  const OctCell bcell = OctreeUtils::FindLeaf(convert_intn(b),
                                                 octree, resln);
  vector<OctreeUtils::CellIntersection> local_intersections;
  vector<OctreeUtils::CellIntersection> all_intersections;
  bool done = false;
  int count = 0;
  OctCell cell =
      OctreeUtils::FindLeaf(convert_intn(cur.p), octree, resln);
  do {
    ++count;
    if (count > 10000)
      throw logic_error("Infinite loop");
    v(cell, a, b, octree, resln, data);

    const vector<OctreeUtils::CellIntersection> local_intersections =
        OctreeUtils::FindIntersections(a, b, cell, /*octree,*/ resln);
    for (const OctreeUtils::CellIntersection& i : local_intersections) {
      if (i.t > cur.t) {
        cur = i;
      }
    }
    for (int i = 0; i < DIM; ++i) {
      const int end = cell.get_origin().s[i];
      if (cur.p.s[i] == end && cur.p.s[i] != 0 && dir[i] == -1) {
        --cur.p.s[i];
      }
    }

    OctCell new_cell = OctreeUtils::FindLeaf(
        convert_intn(cur.p), octree, resln);
    done = local_intersections.empty() ||
        (cell.get_origin() == bcell.get_origin()) ||
        (new_cell.get_origin() == cell.get_origin());
    cell = new_cell;
  } while (!done);
}

struct MCData {
  MCData(vector<CellIntersections>& labels_) : labels(labels_) {}
  vector<CellIntersections>& labels;
  int cur_label;
};

void write_seg(const floatn& a, const floatn& b, const string& fn) {
  using namespace std;
  ofstream out(fn);
  out << std::fixed << a << endl << b << endl;
  out.close();
}

void write_seg(const FloatSegment& s, const string& fn) {
  write_seg(s.a(), s.b(), fn);
}

void write_cell(const intn& o, const int& w, const string& fn) {
  ofstream out(fn);
  out << o << endl;
  out << o+make_intn(w, 0) << endl;
  out << o+make_intn(w, w) << endl;
  out << o+make_intn(0, w) << endl;
  out << o << endl;
  out.close();
}

void error_log(OctCell cell, const floatn& a, const floatn& b,
               const vector<OctNode>& octree, const Resln& resln) {
  cerr << "a = " << std::fixed << a << endl;
  cerr << "b = " << std::fixed << b << endl;
  cerr << "cell = " << cell << endl;
  //cerr << "resln = " << resln << endl;
  ofstream out("mccallback.err");
  //out << cell << " " << a << " " << b << " " << octree << " " << resln << endl;
  out.close();
  write_seg(a, b, "data1.dat");
  const intn o = cell.get_origin();
  const int w = cell.get_width();
  write_cell(o, w, "data2.dat");
  const int w2 = w/2;
  // Boundary
  write_cell(o-make_intn(w2, w2), 2*w, "data3.dat");
}

inline bool within(const float& a, const float& b, const float& epsilon) {
  return fabs(a-b) < epsilon;
}

// Multi-cell callback. Finds intersections with the current cell and
// stores results in data.
void MCCallback(OctCell cell, const floatn& a, const floatn& b,
                  const vector<OctNode>& octree, const Resln& resln,
                  void* data) {
  // const static float EPSILON = 1e-6;

  using namespace Karras;
  MCData* d = static_cast<MCData*>(data);
  const int octant = cell.get_octant();

  // // debug
  // OctCell test_cell;
  // stringstream ss("8796416 6561024 256 22009 3 -1");
  // ss >> test_cell;
  // if (cell == test_cell || cell == fnode) {
  //   cout << "here in MCCallback" << endl;
  // }

  // Get the intersections between the segment and octree cell.
  const vector<OctreeUtils::CellIntersection> intersections =
      OctreeUtils::FindIntersections(a, b, cell, /*octree,*/ resln);
  if (intersections.size() > 2) {
    error_log(cell, a, b, octree, resln);
    cerr << "More than 2 intersections with a cell" << endl;
    for (const OctreeUtils::CellIntersection& ci : intersections) {
      cerr << "  " << ci.p << endl;
    }
    throw logic_error("More than 2 intersections with a cell");
  }
  if (intersections.size() == 2) {
    FloatSegment seg(intersections[0].p,
                  intersections[1].p);
    CellIntersections& cell_intersections = d->labels[cell.get_parent_idx()];
    cell_intersections.set(octant, d->cur_label, seg);
  } else if (intersections.size() == 1) {
    // Find which endpoint is in cell
    const BoundingBox<intn> bb = cell.bb();
    const floatn p = intersections[0].p;
    floatn q;
    bool degenerate = false;
    // if (bb.in_half_open(convert_intn(a), EPSILON)) {
    if (bb.in_half_open(convert_intn(a))) {
      q = a;
    // } else if (bb.in_half_open(convert_intn(b), EPSILON)) {
    } else if (bb.in_half_open(convert_intn(b))) {
      q = b;
    } else if ((within(p.x, bb.min().x, EPSILON) ||
                within(p.x, bb.max().x, EPSILON)) &&
               (within(p.y, bb.min().y, EPSILON) ||
                within(p.y, bb.max().y, EPSILON))) {
      degenerate = true;
    } else {
      degenerate = true;
      // error_log(cell, a, b, octree, resln);
      // throw logic_error("Only one intersection but neither endpoint is"
      //                   " in bounding box");
    }
    if (!degenerate) {
      FloatSegment seg(p, q);
      CellIntersections& cell_intersections = d->labels[cell.get_parent_idx()];
      cell_intersections.set(octant, d->cur_label, seg);
    }
  }
}

void Octree2::FindMultiCells(const Polylines& lines) {
  using namespace Karras;

  cell_intersections.clear();
  cell_intersections.resize(octree.size(), CellIntersections());
  MCData data(cell_intersections);

  const vector<vector<float2>>& polygons = lines.getPolygons();;

  // For each line segment in each polygon do a cell walk, updating
  // the cell_intersections structure.
  intersections.clear();
  for (int j = 0; j < polygons.size(); ++j) {
    const std::vector<float2>& polygon = polygons[j];
    data.cur_label = j;
    for (int i = 0; i < polygon.size() - 1; ++i) {
      const floatn a = obj2Oct(polygon[i]);
      const floatn b = obj2Oct(polygon[i+1]);
      // Visit each octree cell intersected by segment a-b. The visitor
      // is MCCallback.
      CellWalk(a, b, octree, resln, MCCallback, &data);
      vector<OctreeUtils::CellIntersection> local = Walk(a, b);
      for (const OctreeUtils::CellIntersection& ci : local) {
        intersections.push_back(ci.p);
      }
    }
  }

  // Find points that we want to add in order to run a more effective
  // Karras octree construction.
  extra_qpoints.clear();
  _origins.clear();
  _lengths.clear();
  for (int i = 0; i < octree.size(); ++i) {
    const CellIntersections& intersection = cell_intersections[i];
    for (int octant = 0; octant < (1<<DIM); ++octant) {
      const int num_labels = intersection.num_labels(octant);
      // cout << "cell = " << i << "/" << octant << ": " << num_labels << endl;
      // if (intersection.is_multi(octant)) {
      //   {
      for (int j = 0; j < num_labels; ++j) {
      // for (int j = 0; j < 1; ++j) {
        for (int k = j+1; k < num_labels; ++k) {
        // for (int k = j+1; k < std::min(2, num_labels); ++k) {
          // We'll compare segments at j and k.
          FloatSegment segs[2] = { intersection.seg(j, octant),
                                intersection.seg(k, octant) };
          vector<floatn> samples;
          vector<floatn> origins;
          vector<float> lengths;
          if (!Geom::multi_intersection(segs[0], segs[1])) {
            try {
              if (i == fnode.get_parent_idx() && octant == fnode.get_octant()) {
                cout << "here in FitBoxes" << endl;
                write_seg(segs[0], "seg0.dat");
                write_seg(segs[1], "seg1.dat");
              }
              Geom::FitBoxes(segs[0], segs[1], 1, &samples, &origins, &lengths);
            } catch(logic_error& e) {
              cerr << "segments: " << segs[0] << " " << segs[1] << endl;
              cerr << "labels: " << intersection.label(0, octant)
                   << " " << intersection.label(1, octant) << endl;
              write_seg(segs[0], "seg0.dat");
              write_seg(segs[1], "seg1.dat");
              throw e;
            }
          }
          _origins.insert(_origins.end(), origins.begin(), origins.end());
          _lengths.insert(_lengths.end(), lengths.begin(), lengths.end());
          for (const floatn& sample : samples) {
            extra_qpoints.push_back(convert_intn(sample));
          }
        }
      }
    }
  }
}

// void WalkCallback(Karras::OctCell cell, const intn& a, const intn& b,
void WalkCallback(OctCell cell, const floatn& a, const floatn& b,
                  const vector<OctNode>& octree, const Resln& resln,
                  void* data) {
  
  using namespace Karras;
  using OctreeUtils::CellIntersection;
  using OctreeUtils::FindIntersections;

  vector<CellIntersection>& all_intersections =
      *static_cast<vector<CellIntersection>*>(data);

  const vector<CellIntersection> local_intersections =
      FindIntersections(a, b, cell, /*octree,*/ resln);
  for (const CellIntersection& i : local_intersections) {
    // const intn p = i.p;
    all_intersections.push_back(i);
  }
}

vector<OctreeUtils::CellIntersection> Octree2::Walk(
    // const intn& a, const intn& b) {
    const floatn& a, const floatn& b) {
  using namespace Karras;

  vector<OctreeUtils::CellIntersection> all_intersections;
  CellWalk(a, b, octree, resln, WalkCallback, &all_intersections);

  return all_intersections;
}

void Octree2::buildOctVertices() {
  // numVertices = 4;
  // glm::vec3 drawVertices[n];
  // drawVertices[0] = toVec3(oct2Obj(make_int2(0, 0)));
  // drawVertices[1] = toVec3(oct2Obj(make_int2(resln.width, 0)));
  // drawVertices[2] = toVec3(oct2Obj(make_int2(resln.width, resln.width)));
  // drawVertices[3] = toVec3(oct2Obj(make_int2(0, resln.width)));
  vertices.clear();
  vertices.push_back(toVec3(oct2Obj(make_int2(0, 0))));
  vertices.push_back(toVec3(oct2Obj(make_int2(resln.width, 0))));
  vertices.push_back(toVec3(oct2Obj(make_int2(resln.width, 0))));
  vertices.push_back(toVec3(oct2Obj(make_int2(resln.width, resln.width))));
  vertices.push_back(toVec3(oct2Obj(make_int2(resln.width, resln.width))));
  vertices.push_back(toVec3(oct2Obj(make_int2(0, resln.width))));
  vertices.push_back(toVec3(oct2Obj(make_int2(0, resln.width))));
  vertices.push_back(toVec3(oct2Obj(make_int2(0, 0))));

  if (!octree.empty()) {
    drawNode(octree[0], 0, make_intn(0), resln.width);
  }

  glm::vec3* drawVertices = new glm::vec3[vertices.size()];
  std::copy(vertices.begin(), vertices.end(), drawVertices);
  glBindBuffer(GL_ARRAY_BUFFER, drawVertices_vbo);
  glBufferData(
      GL_ARRAY_BUFFER, vertices.size()*sizeof(glm::vec3), drawVertices,
      GL_STATIC_DRAW);
  delete [] drawVertices;

  numIndices = 5;
  drawIndices[0] = 0;
  drawIndices[1] = 1;
  drawIndices[2] = 2;
  drawIndices[3] = 3;
  drawIndices[4] = 0;
  glGenBuffers(1, &drawIndices_vbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawIndices_vbo);
  glBufferData(
      GL_ELEMENT_ARRAY_BUFFER, numIndices*sizeof(GLuint), drawIndices,
      GL_STATIC_DRAW);
}

// length is the length of the parent
void Octree2::drawNode(
    const OctNode& parent, const int parent_idx,
     const intn origin, const int length) {

  for (int i = 0; i < 4; ++i) {
    intn o = origin;
    if (i % 2 == 1)
      o += make_intn(length/2, 0, 0);
    if (i / 2 == 1)
      o += make_intn(0, length/2, 0);
    if (!::is_leaf(&parent, i)) {
      drawNode(octree[parent[i]], parent[i], o, length/2);
    } else {
      // if (&parent == fnode.get_parent() && i == fnode.get_octant()) {
      //   // glLineWidth(3);
      //   // glColor3d(0.5, 0, 0.5);
      //   // glSquare(oct2Obj(o), oct2Obj(length/2));
      // }
//      if (cell_intersections[parent_idx].is_multi(i)) {
//        // glLineWidth(5);
//        // glColor3d(1, 0, 0);
//      } else {
//        // glLineWidth(1);
//        // glColor3dv(octree_color.s);
//      }
      
      // glSquare(oct2Obj(o), oct2Obj(length/2));
      // if (show_vertex_ids) {
      //   stringstream ss;
      //   // ss << o;
      //   ss << parent_idx << "/" << i << " -- " << o;
      //   // ss << o << " " << w << " " << parent_idx << " " << i << " -1";
      //   BitmapString(ss.str(), Oct2Obj(o), 1, 3);
      // }
    }
  }
  // Vertical line
  vertices.push_back(toVec3(oct2Obj(origin+make_intn(length/2, 0))));
  vertices.push_back(toVec3(oct2Obj(origin+make_intn(length/2, length))));
  // Horizontal line
  vertices.push_back(toVec3(oct2Obj(origin+make_intn(0, length/2))));
  vertices.push_back(toVec3(oct2Obj(origin+make_intn(length, length/2))));
}

// void OctViewer2::DrawOctree() const {
//   // cout << "DrawOctree " << octree.size() << endl;

//   glLineWidth(1.0);
//   glColor3dv(octree_color.s);

//   if (!octree.empty()) {
//     glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
//     DrawNode(octree[0], 0, make_intn(0), resln.width);
//     glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//   }

//   // // Draw intersections
//   // glColor3f(0.7, 0.0, 0.7);
//   // for (const intn& i : intersections) {
//   //   glSquareCentered(Oct2Obj(i), Win2Obj(7));
//   // }

//   // // Draw line segment
//   // glColor3f(0.7, 0.0, 0.7);
//   // glBegin(GL_LINES);
//   // glVertex3fv(Oct2Obj(seg_a).s);
//   // glVertex3fv(Oct2Obj(seg_b).s);
//   // glEnd();
// }

void Octree2::render(LinesProgram* program) {
  using namespace std;

  if (numIndices == 0)
    return;

  program->useProgram();

  // program->setMatrix(glm::mat4(1.0));

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, drawVertices_vbo);

  glVertexAttribPointer(
      program->getVertexLoc(), 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(program->getVertexLoc());

  program->setColor(make_float3(0.0, 0.0, 0.0));

  // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawIndices_vbo);
  // glDrawElements(GL_LINE_STRIP, numIndices, GL_UNSIGNED_INT, (void*)0);
  glLineWidth(1.0);
  glDrawArrays(GL_LINES, 0, vertices.size());

  print_error("Octree2::render 1");
}
