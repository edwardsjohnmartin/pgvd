#include <iostream>
#include <fstream>
#include <sstream>

#include <cstdio>

#include "../GLUtilities/gl_utils.h"
#include "./Octree2.h"
#include "../Timer/timer.h"

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
    //Probably should be parallelized...
    floatn minimum = karras_points[0];
    floatn maximum = karras_points[0];
    for (int i = 1; i < totalPoints; ++i) {
        minimum = min_floatn(karras_points[i], minimum);
        maximum = max_floatn(karras_points[i], maximum);
    }
    bb = BB_initialize(&minimum, &maximum);
    bb = BB_make_centered_square(&bb);
}

void Octree2::quantizePoints(int numResolutionPoints) {
    Kernels::QuantizePoints_p(totalPoints, karrasPointsBuffer, quantizedPointsBuffer, bb.minimum, resln.width, bb.maxwidth);
}

void Octree2::makeZOrderPoints() {
    //Timer t("Making Z Order points.");

    Kernels::PointsToMorton_p(quantizedPointsBuffer, zpoints, totalPoints, resln.bits);

    //CLFW::DefaultQueue.finish();
}

void Octree2::GetUnorderedBCellFacetPairs() {

    Kernels::GetBCellLCP_p(linesBuffer, zpoints, BCells,
        unorderedLineIndices, lines.size(), resln.mbits);
    
}

void Octree2::buildVertexOctree() {
    Kernels::BuildOctree_p(zpoints, totalPoints, octreeSize, resln.bits, resln.mbits);
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
    //Timer t("Getting resolution points");
    if (lines.size() < 2) return;
    resolutionPoints.resize(0);

    resolutionPointsSize = 0;
    //Parallel version
    cl::Buffer conflictInfoBuffer, resolutionCounts, resolutionPredicates, scannedCounts;
    assert(Kernels::GetResolutionPointsInfo_p(octreeSize, conflictsBuffer, linesBuffer,
        quantizedPointsBuffer, conflictInfoBuffer, resolutionCounts, resolutionPredicates) == CL_SUCCESS);

    vector<int> test(1);
    CLFW::DefaultQueue.enqueueReadBuffer(resolutionCounts, CL_TRUE, 0, sizeof(cl_int), test.data());
    // cout << test[0] << " =? " << sizeof(ConflictInfo) << endl;
    // cout << "Test = " << test[0] << endl;



    assert(CLFW::get(scannedCounts, "sResCnts", Kernels::nextPow2(octreeSize * 4) * sizeof(cl_int)) == CL_SUCCESS);
    assert(Kernels::StreamScan_p(resolutionCounts, scannedCounts, Kernels::nextPow2(4 * octreeSize),
        "resolutionIntermediate", false) == CL_SUCCESS);
    assert(CLFW::DefaultQueue.enqueueReadBuffer(scannedCounts, CL_TRUE,
        (octreeSize * 4 * sizeof(cl_int)) - sizeof(cl_int), sizeof(cl_int),
        &resolutionPointsSize) == CL_SUCCESS);

    if (resolutionPointsSize < 0) {
        /*using namespace Kernels;
        cout << "warning: additional total " << resolutionPointsSize << endl;

        vector<BigUnsigned> before, after;
        vector<intn> qpoints;
        vector<floatn> karrasPoints;
        vector<int> counts, scannedCounts_vec;
        vector<Conflict> conflicts_vec;
        DownloadFloatnPoints(karrasPoints, CLFW::Buffers["karrasPointsBuffer"], karras_points.size());
        DownloadQPoints(qpoints, quantizedPointsBuffer, karras_points.size());
        DownloadZPoints(before, zpoints, karras_points.size());
        DownloadInts(resolutionCounts, counts, octreeSize * 4);
        DownloadInts(scannedCounts, scannedCounts_vec, octreeSize * 4);
        DownloadConflicts(conflicts_vec, conflictsBuffer, octreeSize * 4);
        cout << "karraspoints" << endl;
        for (int i = 0; i < karrasPoints.size(); ++i) {
            cout << i << " " << karrasPoints[i] << endl;
        }
        cout << "qPoints" << endl;
        for (int i = 0; i < qpoints.size(); ++i) {
            cout << i << " " << qpoints[i] << endl;
        }
        cout << "zPoints" << endl;
        for (int i = 0; i < before.size(); ++i) {
            cout << i << " " << buToString(before[i]) << endl;
        }
        cout << "Conflicts" << endl;
        for (int i = 0; i < conflicts_vec.size(); ++i) {
            cout << i << " " << conflicts[i] << endl;
        }
        cout << "Counts" << endl;
        for (int i = 0; i < counts.size(); ++i) {
            cout << i << " " << counts[i] << endl;
        }
        cout << "Scanned Counts" << endl;
        for (int i = 0; i < scannedCounts_vec.size(); ++i) {
            cout << i << " " << scannedCounts_vec[i] << endl;
        }
        CLFW::get(scannedCounts, "sResCnts", Kernels::nextPow2(octreeSize * 4) * sizeof(cl_int));
        Kernels::StreamScan_p(resolutionCounts, scannedCounts, Kernels::nextPow2(4 * octreeSize),
            "resolutionIntermediate", false);
        CLFW::DefaultQueue.enqueueReadBuffer(scannedCounts, CL_TRUE,
            (octreeSize * 4 * sizeof(cl_int)) - sizeof(cl_int), sizeof(cl_int),
            &resolutionPointsSize);*/
        return;
    }

    assert(Kernels::GetResolutionPoints_p(octreeSize, resolutionPointsSize, conflictsBuffer, linesBuffer,
        quantizedPointsBuffer, conflictInfoBuffer, scannedCounts, resolutionPredicates,
        resolutionPointsBuffer) == CL_SUCCESS);

    //if (resolutionPointsSize < 0) {
    //    cout << "parallel" << endl;
    //    vector<int> testVec(4 * octreeSize);
    //    CLFW::DefaultQueue.enqueueReadBuffer(resolutionCounts, CL_TRUE, 0, sizeof(cl_int) * 4 * octreeSize, testVec.data());
    //    for (int i = 0; i < testVec.size(); ++i) {
    //        cout << i << " " << testVec[i] << endl;
    //    }

    //    cout << "serial" << endl;
    //    int totalAdditionalPoints = 0;
    //    //unsigned int resolutionPointsSize = 0;
    //    Kernels::DownloadQPoints(quantized_points, CLFW::Buffers["qPoints"], totalPoints);
    //    Kernels::DownloadConflicts(conflicts, conflictsBuffer, 4 * octreeSize);
    //    vector<int> testcounts(octreeSize * 4);
    //    Kernels::SampleConflictCounts_s(octreeSize, conflicts.data(), &totalAdditionalPoints, testcounts, orderedLines.data(),
    //        quantized_points.data(), resolutionPoints);
    //    for (int i = 0; i < testcounts.size(); ++i) {
    //        //karras_points.push_back(UnquantizePoint(&resolutionPoints[i], &bb.minimum, resln.width, bb.maxwidth));
    //        cout <<i<<" "<< testcounts[i] << endl;
    //    }
    //}

    //vector<intn> gpuResolutionPoints(resolutionPointsSize);
    //CLFW::DefaultQueue.enqueueReadBuffer(resolutionPointsBuffer, CL_TRUE, 0, sizeof(intn)*resolutionPointsSize, gpuResolutionPoints.data());
    //assert(CLFW::DefaultQueue.enqueueReadBuffer(resolutionPointsBuffer, CL_TRUE, 0, 
    //resolutionPointsSize * sizeof(intn), gpuResolutionPoints.data())==CL_SUCCESS);

    ////Tests
    //vector<int> testCounts(4 * octreeSize);
    //cout << "total additional points " <<" = " << totalAdditionalPoints << endl;

    //Total points by both must match
    //if (resolutionPointsSize != totalAdditionalPoints) 
      //  cout<<"Warning, GPU additional points count does not match CPU addition points count"<<endl;  

    ////Each resolution point must match.
    //for (int i = 0; i < resolutionPoints.size(); i++) {
    //    cout << "res point " << i <<": " << resolutionPoints[i] <<  " vs " << gpuResolutionPoints[i] << endl;
    //    assert(gpuResolutionPoints[i] == resolutionPoints[i]);
    //}

    //using namespace GLUtilities;
    //for (int i = 0; i < gpuResolutionPoints.size(); ++i) {
    //    intn q = gpuResolutionPoints[i];
    //    floatn newp = UnquantizePoint(&q, &bb.minimum, resln.width, bb.maxwidth);

    //    resolutionPoints.push_back(q);

    //    GLUtilities::Point p = { { newp.x, newp.y , 0.0, 1.0 },{ 1.0,0.0,0.0,1.0 } };
    //    Sketcher::instance()->add(p);
    //}
    //CLFW::DefaultQueue.finish();
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
        assert(error == CL_SUCCESS);
    }

    //CLFW::DefaultQueue.finish();
}

void Octree2::build(const PolyLines *polyLines) {
    Timer t("overall");
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

    printf("Starting build\n");
    Timer t("Build octree");
    totalPoints = karras_points.size();
    computeBoundingBox(totalPoints);
    UploadKarrasPoints(karras_points, karrasPointsBuffer);
    UploadLines(lines, linesBuffer);
    quantizePoints();
    makeZOrderPoints();
    do {
        totalIterations++;
        CLFW::DefaultQueue = CLFW::Queues[0];
        previousSize = octreeSize;

        insertResolutionPoints();
        makeZOrderPoints(); //TODO: only z-order the additional points here...
        CLFW::DefaultQueue.finish();

        /*
          On one queue, sort the lines for the ambiguous cell detection.
          On the other, start building the karras octree.
        */
        CLFW::DefaultQueue = CLFW::Queues[1];
        GetUnorderedBCellFacetPairs(); //13% (concurrent)
        CLFW::DefaultQueue = CLFW::Queues[0];
        buildVertexOctree(); //40% (concurrent)
        CLFW::Queues[0].finish();
        CLFW::Queues[1].finish();


        /* Identify the cells that contain more than one object. */
        if (lines.size() == 0 || lines.size() == 1) break;
        identifyConflictCells(); //43%

        /* Resolve one iteration of conflicts. */
        getResolutionPoints();//12%
    } while (previousSize != octreeSize && resolutionPointsSize != 0);

    cl::Buffer octreeBuffer = CLFW::Buffers["octree"];
    octree.resize(octreeSize);
    cl_int error = CLFW::DefaultQueue.enqueueReadBuffer(octreeBuffer, CL_TRUE, 0, sizeof(OctNode)*octreeSize, octree.data());
    assert(error == CL_SUCCESS);

    t.restart("Rendering stuff");

    /* Add the octnodes so they'll be rendered with OpenGL. */
    addOctreeNodes();
    /* Add conflict cells so they'll be rendered with OpenGL. */
    addConflictCells();

    // cout << "Total iterations: " << totalIterations << endl;
    cout << "Octree size: " << octree.size() << endl;
}

/* Drawing Methods */
void Octree2::addOctreeNodes() {
    floatn temp;
    floatn center;

    if (octreeSize == 0) return;
    center = (bb.minimum + bb.maxwidth*.5);
    float3 color = { 1.0, 1.0, 1.0 };
    addOctreeNodes(0, center, bb.maxwidth, color);
}

void Octree2::addOctreeNodes(int index, floatn offset, float scale, float3 color) {
    Instance i = { offset.x, offset.y, 0.0, scale, color.x, color.y, color.z };
    gl_instances.push_back(i);
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
    if (conflicts.size() == 0) return;
    for (int i = 0; i < octreeSize; ++i) {
        for (int j = 0; j < 4; j++) {
            if (conflicts[4 * i + j].color == -2) {
                addLeaf(i, j, { 1.0, 0.0, 0.0 });
            }
        }
    }
}

void Octree2::draw() {
    Shaders::boxProgram->use();
    print_gl_error();
    glBindVertexArray(boxProgram_vao);
    print_gl_error();
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
    print_gl_error();
    glBufferData(GL_ARRAY_BUFFER, sizeof(Instance) * gl_instances.size(), gl_instances.data(), GL_STREAM_DRAW);
    print_gl_error();
    glm::mat4 identity(1.0);
    glUniformMatrix4fv(Shaders::boxProgram->matrix_id, 1, 0, &(identity[0].x)); //glm::value_ptr wont work on identity for some reason...
    glUniform1f(Shaders::boxProgram->pointSize_id, 10.0);
    print_gl_error();
    glLineWidth(2.0);
    ignore_gl_error();
    glDrawElementsInstanced(GL_LINES, 12 * 2, GL_UNSIGNED_BYTE, 0, gl_instances.size());
    print_gl_error();
    glBindVertexArray(0);
    print_gl_error();
}
