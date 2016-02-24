#ifndef __LINES_H__
#define __LINES_H__

#include <vector>

#include "./glm/gtc/matrix_transform.hpp"

#include "../opencl/vec.h"
#include "./LinesProgram.h"
#include "../Options.h"

class Polylines {
 private:
  int capacity;
  int size;
  glm::vec3* points;
  // Index of last+1 point in a line
  std::vector<int> lasts;

  GLuint pointsVboId;
  GLuint pointsVaoId;

 public:
  Polylines() : capacity(1024), size(0), points(new glm::vec3[capacity]) {
    glGenBuffers(1, &pointsVboId);
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);
    glBufferData(
        GL_ARRAY_BUFFER, capacity*sizeof(glm::vec3), points, GL_STATIC_DRAW);
	
    glGenVertexArrays(1, &pointsVaoId);
    glBindVertexArray(pointsVaoId);
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);
  }

  ~Polylines() {
    delete [] points;
  }

  void clear() {
    size = 0;
    lasts.clear();
  }

  void addPoint(const float2& p) {
    using namespace std;
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);

    if (size == capacity) {
      glm::vec3* temp = new glm::vec3[capacity*2];
      memcpy(temp, points, capacity*sizeof(glm::vec3));
      delete [] points;
      points = temp;
      capacity *= 2;
      glBufferData(
          GL_ARRAY_BUFFER, capacity*sizeof(glm::vec3), points, GL_STATIC_DRAW);
      cout << "Updating capacity to " << capacity << endl;
    }
    points[size] = glm::vec3(p.x, p.y, 0.0);

    glBufferSubData(
        GL_ARRAY_BUFFER, size*sizeof(glm::vec3),
        sizeof(glm::vec3), points+size);

    ++size;
    lasts.back() = size;
  }

  // This is a test method for the fit program. It is not for general use.
  void replacePoint(const float2& p, const int lineIdx, const int vertexIdx) {
    using namespace std;
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);

    int i = (lineIdx == 0) ? vertexIdx + 0 : vertexIdx + 2;
    points[i] = glm::vec3(p.x, p.y, 0.0);

    glBufferSubData(
        GL_ARRAY_BUFFER, i*sizeof(glm::vec3),
        sizeof(glm::vec3), points+i);
  }

  void newLine(const float2& p) {
    lasts.push_back(0);
    addPoint(p);
  }

  std::vector<std::vector<float2>> getPolygons() const {
    using namespace std;
    vector<vector<float2>> ret;

    int first = 0;
    for (int i = 0; i < lasts.size(); ++i) {
      const int last = lasts[i];
      vector<float2> polygon;
      for (int j = first; j < last; ++j) {
        const glm::vec3& p = points[j];
        polygon.push_back(make_float2(p.x, p.y));
      }
      if (polygon.size() > 1) {
        ret.push_back(polygon);
      }
      first = lasts[i];
    }
    return ret;
  }

  void render(LinesProgram* program) {
    using namespace std;

    program->useProgram();
    // glm::mat4 matrix = glm::mat4(1.0);
    // program->setMatrix(matrix);
		
    program->setPointSize(5.0);

    glBindVertexArray(pointsVaoId);
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);

    glVertexAttribPointer(
        program->getVertexLoc(), 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(program->getVertexLoc());

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    int first = 0;
    for (int i = 0; i < lasts.size(); ++i) {
      const int len = lasts[i]-first;
      program->setColor(i);
      if (options.showObjectVertices) {
        glDrawArrays(GL_POINTS, first, len);
      }
      if (options.showObjects) {
        glDrawArrays(GL_LINE_STRIP, first, len);
      }
      first = lasts[i];
    }
    print_error("Polylines");
  }
};

#endif
