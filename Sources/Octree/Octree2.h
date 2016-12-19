#ifndef __OCTREE_2_H__
#define __OCTREE_2_H__

#include "../Timer/timer.h"
#include "clfw.hpp"
#include "../Shaders/Shaders.hpp"
#include "../Polylines/Polylines.h"
#include "../GLUtilities/gl_utils.h"
#include "../Options/Options.h"
#include "../Kernels/Kernels.h"
#include "../../SharedSources/BoundingBox/BoundingBox.h"
#include <glm/glm.hpp>

extern "C" {
#include "../../SharedSources/OctreeResolution/Resln.h"
#include "../../SharedSources/Line/Line.h"
}
#include "../../SharedSources/Octree/OctNode.h"
#include "../../SharedSources/Quantize/Quantize.h"
#include "../../Sources/GLUtilities/Sketcher.h"

class Octree2 {
private:
    std::vector<OctNode> octree;
    std::vector<floatn> karras_points;
    std::vector<intn> quantized_points;
    std::vector<intn> resolutionPoints;
    std::vector<Line> lines;
    std::vector<Line> orderedLines;
    std::vector<Conflict> conflicts;
    std::vector<Leaf> leaves;
    int octreeSize;
    BoundingBox bb;
    Resln resln;
    int totalPoints;
    int totalLeaves;
    int resolutionPointsSize;

    std::vector<glm::vec3> offsets;
    std::vector<glm::vec3> colors;
    std::vector<float> scales;

   /* GLuint drawVertices_vbo;
    GLuint drawIndices_vbo;
    GLuint vao;*/

    GLuint boxProgram_vao;
    GLuint positions_vbo;
    GLuint position_indices_vbo;
    GLuint instance_vbo;
    cl::Buffer quantizedPointsBuffer;
    cl::Buffer karrasPointsBuffer;
    cl::Buffer zpoints;
    cl::Buffer zpointsCopy;
    cl::Buffer linesBuffer;
    cl::Buffer BCells;
    cl::Buffer unorderedLineIndices;
    cl::Buffer unorderedNodeIndices;
    cl::Buffer orderedLineIndices;
    cl::Buffer orderedNodeIndices;
    cl::Buffer conflictsBuffer;
    cl::Buffer resolutionPointsBuffer;
    cl::Buffer leavesBuffer;

    void getPoints(const PolyLines *polyLines);
    void getBoundingBox(const int totalPoints);
    void getQuantizedPoints(int numResolutionPoints = 0);
    void getZOrderPoints();
    void getUnorderedBCellFacetPairs();
    void getVertexOctree();
    void getConflictCells();
    void getResolutionPoints();
    void addResolutionPoints();

public:
    Octree2();
    void build(const PolyLines* lines);
    typedef struct {
        float offset[3];
        float scale;
        float color[3];
    } Instance;
    std::vector<Instance> gl_instances;

    /* Drawing Methods */
    void draw(const glm::mat4& mvMatrix);

private:

    /* Drawing Methods */
    void addOctreeNodes();
    void addOctreeNodes(int index, floatn offset, float scale, float3 color);
    void addLeaf(int internalIndex, int leafIndex, float3 color);
    void addConflictCells();
    void drawResolutionPoints();
};


#endif
