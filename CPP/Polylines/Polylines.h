#ifndef __LINES_H__
#define __LINES_H__

#include <vector>

#include "./glm/gtc/matrix_transform.hpp"
extern "C" {
  #include "../../C/Line/Line.h"
}
#include "../../C/Vector/vec_n.h"
#include "../../CPP/Shaders/Shaders.hpp"
#include "../Options/Options.h"
#include "../Color/Color.h"
#include <cstring>
#include <iostream>

class PolyLines {
 private:
  int capacity;
  int size;
  float_3* points;
  // Index of last+1 point in a line
  std::vector<int> lasts;
  std::vector<float_3> colors;

  GLuint pointsVboId;
  GLuint pointsVaoId;

 public:
  PolyLines() : capacity(1024), size(0), points(new float_3[capacity]) {
    glGenBuffers(1, &pointsVboId);
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);
    glBufferData(GL_ARRAY_BUFFER, capacity*sizeof(float_3), points, GL_STATIC_DRAW);
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

  void addPoint(const float_2& p) {
    using namespace std;
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);

    if (size == capacity) {
      float_3* temp = new float_3[capacity*2];
      memcpy(temp, points, capacity*sizeof(float_3));
      delete [] points;
      points = temp;
      capacity *= 2;
      glBufferData(
          GL_ARRAY_BUFFER, capacity*sizeof(float_3), points, GL_STATIC_DRAW);
      cout << "Updating capacity to " << capacity << endl;
    }
		X_(points[size]) = X_(p);
		Y_(points[size]) = Y_(p);
		Z_(points[size]) = 0.0;

    glBufferSubData(
        GL_ARRAY_BUFFER, size*sizeof(float_3),
        sizeof(float_3), points+size);

    ++size;
    lasts.back() = size;
  }

  // This is a test method for the fit program. It is not for general use.
  void replacePoint(const float_2& p, const int lineIdx, const int vertexIdx) {
    using namespace std;
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);

    int i = (lineIdx == 0) ? vertexIdx + 0 : vertexIdx + 2;
		X_(points[i]) = X_(p);
		Y_(points[i]) = Y_(p);
		Z_(points[i]) = 0.0;

    glBufferSubData(
        GL_ARRAY_BUFFER, i*sizeof(float_3),
        sizeof(float_3), points+i);
  }

  void setPoint(const float_2& p, bool first) {
    int i = first ? 0 : 1;
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId);
		X_(points[i]) = X_(p);
		Y_(points[i]) = Y_(p);
		Z_(points[i]) = 0.0;
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(float_3)*i, sizeof(float_3), points + i);
  }

  void newLine(const float_2& p) {
    lasts.push_back(0);
    colors.push_back(Color::randomColor());
    addPoint(p);
  }

  std::vector<std::vector<float_2>> getPolygons() const {
    using namespace std;
    vector<vector<float_2>> ret;

    int first = 0;
    for (int i = 0; i < lasts.size(); ++i) {
      const int last = lasts[i];
      vector<float_2> polygon;
      for (int j = first; j < last; ++j) {
        float_3 p = points[j];
        polygon.push_back({X_(p), Y_(p)});
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

  void render() {
    using namespace std;
    using namespace GLUtilities;
    Shaders::lineProgram->use();
    glUniform1f(Shaders::lineProgram->pointSize_id, 5.0);

    print_error("Polylines0");

    glBindVertexArray(pointsVaoId);
    glBindBuffer(GL_ARRAY_BUFFER, pointsVboId); //Think this isn't required after vao setup.
    print_error("Polylines0.1");
    glVertexAttribPointer( Shaders::lineProgram->position_id, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    print_error("Polylines0.2");
    glEnableVertexAttribArray(Shaders::lineProgram->position_id);
    print_error("Polylines0.3");

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.0);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
      if (error == GL_INVALID_VALUE) {
        // Line widths of >1 not supported on Mac OS X. No big deal.
      } else {
        print_error(error, "Polylines0.4");
      }
    }

    int first = 0;
    for (int i = 0; i < lasts.size(); ++i) {
      const int len = lasts[i]-first;

      float_3 color = colors[i];
      glUniform3fv(Shaders::lineProgram->color_uniform_id, 1, color.s);

      if (Options::showObjectVertices) {
        glDrawArrays(GL_POINTS, first, len);
        print_error("Polylines1");
      }
      if (Options::showObjects) {
        glDrawArrays(GL_LINE_STRIP, first, len);
        print_error("Polylines2");
      }
      first = lasts[i];
    }
    print_error("Polylines");
  }

  void readFile(const char *filename);
};

#endif
