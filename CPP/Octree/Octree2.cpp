#include <iostream>
#include <fstream>
#include <sstream>

#include <cstdio>

#include "./Octree2.h"
#include "../Karras/Karras.h" //TODO: parallelize quantization

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
      GL_ARRAY_BUFFER, n*sizeof(float_3), drawVertices, GL_STATIC_DRAW);
	
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

  resln = make_resln(1 << Options::max_level);
}

void Octree2::build(const PolyLines *polyLines) {
  using namespace std;
  cl::Buffer quantizedPointsBuffer, karrasPointsBuffer, zpoints, zpointsCopy, sortedLinesBuffer;
  karras_points.clear();
  quantized_points.clear();
  octree.clear();
  gl_instances.clear();

  const vector<vector<float_2>>& polygons = polyLines->getPolygons();
  lines = polyLines->getLines();

  if (polygons.empty()) return;
  
  // Get all vertices into a 1D array (karras_points).
  for (int i = 0; i < polygons.size(); ++i) {
    const vector<float_2>& polygon = polygons[i];
    for (int j = 0; j < polygon.size()-1; ++j) {
      karras_points.push_back(polygon[j]);
    }
    karras_points.push_back(polygon.back());
  }

  //0. Compute bounding box
  //Probably should be parallelized...
  float_n minimum;
  float_n maximum;
  copy_fvfv(&minimum, &karras_points[0]);
  copy_fvfv(&maximum, &karras_points[0]);
  for (int i = 1; i < karras_points.size(); ++i) {
    min_fvfv(&minimum, &minimum, &karras_points[i]);
    max_fvfv(&maximum, &maximum, &karras_points[i]);
  }
  BB_initialize(&bb, &minimum, &maximum);
  BB_make_centered_square(&bb, &bb);
  
  //1. Quantize the cartesian points. MOVE TO OPENCL...
	Kernels::UploadKarrasPoints(karras_points, karrasPointsBuffer);
  quantized_points = Karras::Quantize((float_n*)karras_points.data(), karras_points.size(), resln, &bb);

  //2. Convert points to Z-Order
  CLFW::DefaultQueue = CLFW::Queues[0];
  Kernels::UploadQuantizedPoints(quantized_points, quantizedPointsBuffer);
  Kernels::PointsToMorton_p(quantizedPointsBuffer, zpoints, quantized_points.size(), resln.bits);
  CLFW::DefaultQueue.finish();

  //3A. Sort lines by level, then value.
  CLFW::DefaultQueue = CLFW::Queues[1];
  Kernels::SortLinesByLvlThenVal_p(lines, sortedLinesBuffer, zpoints, resln);
  
  if (quantized_points.size() > 1) {
    //3B. Create a vertex octree with using Karras' algorithm
    CLFW::DefaultQueue = CLFW::Queues[0];
    Kernels::BuildOctree_p(zpoints, quantized_points.size(), octree, resln.bits, resln.mbits);

    assert(CLFW::Queues[0].finish() == CL_SUCCESS);
    assert(CLFW::Queues[1].finish() == CL_SUCCESS);
    addOctreeNodes();

		if (lines.size() > 1) {
			//4. Identify ambiguous cells
			cl::Buffer octreeBuffer = CLFW::Buffers["octree"];
			float_n octreeCenter;
			float octreeWidth;
			BB_max_size(&bb, &octreeWidth);
			BB_center(&bb, &octreeCenter);
			cl::Buffer conflictPairsBuffer;
			Kernels::FindConflictCells_p(sortedLinesBuffer, lines.size(), octreeBuffer, octree.size(), 
				octreeCenter, octreeWidth, conflictPairsBuffer, karrasPointsBuffer);
			vector<ConflictPair> conflictPairs;
			Kernels::DownloadConflictPairs(conflictPairs, conflictPairsBuffer, octree.size() * 4);
			assert(CLFW::Queues[0].finish() == CL_SUCCESS);
    
			if (lines.size() > 1)
				for (int i = 0; i < octree.size(); ++i) {
					for (int j = 0; j < 4; j++) {
						if (conflictPairs[4 * i + j].i[0] == -2) {
							addLeaf(i, j, { 1.0, 0.0, 0.0 });
						}
					}
				}
		}
  }
}

/* Drawing Methods */
void Octree2::addOctreeNodes() {
  float_2 temp;
  float_n center;
  float width;

  if (octree.size() == 0) return;
  BB_max_size(&bb, &width);
  copy_fvf(&temp, width / 2.0F);
  add_fvfv(&center, &bb.minimum, &temp);
  float_3 color = { 1.0, 1.0, 1.0 };
  cout << endl;
  addOctreeNodes(0, center, width, color);
}
void Octree2::addOctreeNodes(int index, float_n offset, float scale, float_3 color) {
  Instance i = { X_(offset), Y_(offset), 0.0, scale, X_(color), Y_(color), Z_(color) };
  gl_instances.push_back(i);
  if (index != -1) {
    OctNode current = octree[index];
    scale /= 2.0;
    float shift = scale / 2.0;
    addOctreeNodes(current.children[0], { X_(offset) - shift, Y_(offset) - shift }, scale, color);
    addOctreeNodes(current.children[1], { X_(offset) + shift, Y_(offset) - shift }, scale, color);
    addOctreeNodes(current.children[2], { X_(offset) - shift, Y_(offset) + shift }, scale, color);
    addOctreeNodes(current.children[3], { X_(offset) + shift, Y_(offset) + shift }, scale, color);
  }
}

void Octree2::addLeaf(int internalIndex, int childIndex, float_3 color) {
  float_n center;
  BB_center(&bb, &center);
  float octreeWidth;
  BB_max_size(&bb, &octreeWidth);
  OctNode node = octree[internalIndex];

  //Shift to account for the leaf
  float width = octreeWidth / (1 << (node.level + 1));
  float shift = width / 2.0;
  X_(center) += (childIndex & 1) ? shift : -shift;
  Y_(center) += (childIndex & 2) ? shift : -shift;

  //Shift for the internal node
  while (node.parent != -1) {
    for (childIndex = 0; childIndex < 4; ++childIndex) {
      if (octree[node.parent].children[childIndex] == internalIndex)
        break;
    }

    shift *= 2.0;
    X_(center) += (childIndex & 1) ? shift : -shift;
    Y_(center) += (childIndex & 2) ? shift : -shift;

    internalIndex = node.parent;
    node = octree[node.parent];
  }

  Instance i = {
    { X_(center), Y_(center), 0.0 }
    , width
    , { X_(color), Y_(color), Z_(color) }
  };
  gl_instances.push_back(i);
}

void Octree2::draw() {
  Shaders::boxProgram->use();
  glBindVertexArray(boxProgram_vao);
  glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * gl_instances.size(), gl_instances.data(), GL_STREAM_DRAW);
  glm::mat4 identity(1.0);
  glUniformMatrix4fv(Shaders::boxProgram->matrix_id, 1, 0, &(identity[0].x)); //glm::value_ptr wont work on identity for some reason...
  glUniform1f(Shaders::boxProgram->pointSize_id, 10.0);
  assert(glGetError() == GL_NO_ERROR);
  glLineWidth( 2.0);
  glDrawElementsInstanced(GL_LINES, 12 * 2, GL_UNSIGNED_BYTE, 0, gl_instances.size());
  glBindVertexArray(0);
}
