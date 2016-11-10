#ifndef __LINES_H__
#define __LINES_H__

#include <vector>

#include "./glm/gtc/matrix_transform.hpp"
extern "C" {
  #include "../../SharedSources/Line/Line.h"
}
#include "../../Sources/Shaders/Shaders.hpp"
#include "../../SharedSources/Vector/vec.h"
#include "../Options/Options.h"
#include "../Color/Color.h"
#include <cstring>
#include <iostream>

class PolyLines {
 private:
  int capacity;
  int size;
  floatn* points;
  // Index of last+1 point in a line
  std::vector<int> lasts;
  std::vector<float3> colors;

  GLuint pointsVboId;
  GLuint pointsVaoId;

 public:
  PolyLines() : capacity(1024), size(0), points(new floatn[capacity]) {
    glGenBuffers(1, &pointsVboId);
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);
    glBufferData(GL_ARRAY_BUFFER, capacity*sizeof(floatn), points, GL_STATIC_DRAW);
    glGenVertexArrays(1, &pointsVaoId);
    glBindVertexArray(pointsVaoId);
    glEnableVertexAttribArray(Shaders::lineProgram->position_id);
    assert(glGetError() == GL_NO_ERROR);
    srand(time(NULL));
  }

  ~PolyLines() {
    delete [] points;
  }

  void clear() {
    size = 0;
    lasts.clear();
  }

  void addPoint(const floatn& p) {
    using namespace std;
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);

    if (size == capacity) {
      floatn* temp = new floatn[capacity*2];
      memcpy(temp, points, capacity*sizeof(floatn));
      delete [] points;
      points = temp;
      capacity *= 2;
      glBufferData(
          GL_ARRAY_BUFFER, capacity*sizeof(floatn), points, GL_STATIC_DRAW);
      cout << "Updating capacity to " << capacity << endl;
    }
    points[size] = p;

    glBufferSubData(
        GL_ARRAY_BUFFER, size*sizeof(floatn),
        sizeof(floatn), points+size);

    ++size;
    lasts.back() = size;
  }

  void newLine(const floatn& p) {
    lasts.push_back(0);
    colors.push_back(Color::randomColor());
    addPoint(p);
  }

  std::vector<std::vector<floatn>> getPolygons() const {
    using namespace std;
    vector<vector<floatn>> ret;

    int first = 0;
    for (int i = 0; i < lasts.size(); ++i) {
      const int last = lasts[i];
      vector<floatn> polygon;
      for (int j = first; j < last; ++j) {
        polygon.push_back(points[j]);
      }
      if (polygon.size() > 1) {
        ret.push_back(polygon);
      }
      first = lasts[i];
    }
    return ret;
  }

  std::vector<Line> getLines() const {
    using namespace std;
    vector<Line> lines;
    int totalSkips = 0; //we skip single point polylines.
    int first = 0;
    //For each polyline 
    for (int i = 0; i < lasts.size(); ++i) {
      int last = lasts[i];
      first = (i != 0) ? lasts[i - 1] : i;
      //if the polyline has more than one point
      if (last - first > 1) {
        //then add the lines in this polyline
        for (int j = 0; j < (last - first) - 1; ++j) {
          Line line;
          line.firstIndex = first + j - totalSkips;
          line.secondIndex = first + j + 1 - totalSkips;
          line.color = first;
          lines.push_back(line);
        }
      }
      else totalSkips++;
    }
    return lines;
  }

  void render(const glm::mat4& mvMatrix) {
    using namespace std;
    using namespace GLUtilities;
    Shaders::lineProgram->use();
    glUniform1f(Shaders::lineProgram->pointSize_id, 5.0);

    print_gl_error();

    glBindVertexArray(pointsVaoId);
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId); //Think this isn't required after vao setup.
    print_gl_error();
    glVertexAttribPointer( Shaders::lineProgram->position_id, sizeof(floatn)/sizeof(cl_float), GL_FLOAT, GL_FALSE, 0, NULL);
    print_gl_error();
    glEnableVertexAttribArray(Shaders::lineProgram->position_id);
    print_gl_error();

    glUniformMatrix4fv(
        Shaders::lineProgram->matrix_id, 1, 0, &(mvMatrix[0].x));

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.0);
    GLenum error = glGetError();
    // if (error != GL_NO_ERROR) {
    //   if (error == GL_INVALID_VALUE) {
    //     // Line widths of >1 not supported on Mac OS X. No big deal.
    //   } else {
    //     print_gl_error(error, "Polylines0.4");
    //   }
    // }

    int first = 0;
    for (int i = 0; i < lasts.size(); ++i) {
      const int len = lasts[i]-first;

      float3 color = colors[i];
      glUniform3fv(Shaders::lineProgram->color_uniform_id, 1, color.s);

      if (Options::showObjectVertices) {
        glDrawArrays(GL_POINTS, first, len);
        print_gl_error();
      }
      if (Options::showObjects) {
        glDrawArrays(GL_LINE_STRIP, first, len);
        print_gl_error();
      }
      first = lasts[i];
    }
    print_gl_error();
  }
};

#endif
