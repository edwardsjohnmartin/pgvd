#ifndef __LINE_PROGRAM_H__
#define __LINE_PROGRAM_H__

#include <assert.h>
#include <iostream>

#include "./glm/mat4x4.hpp"
#include "./glm/gtc/type_ptr.hpp"

#include "../../GLUtilities/gl_utils.h"
#include "../../Color/Color.h"

using namespace GLUtilities;
class LinesProgram {
 private:
  GLuint program;
  GLuint vertexId;
  GLuint matrixId;
  GLuint colorId;
  GLuint pointSizeId;

  glm::mat4 matrix;

 public:
  LinesProgram() {
    //------------------------------------------------------------
    // Initialize shaders
    //------------------------------------------------------------
    char vertex_shader[1024 * 256];
    char fragment_shader[1024 * 256];
    GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    std::string vshaderfn = "./OpenGL/Shaders/lines-v1.2.vert";
    std::string fshaderfn = "./OpenGL/Shaders/lines-v1.2.frag";
    if ((major == 4 && minor >= 1) || major >= 5) {
      vshaderfn = "./OpenGL/Shaders/lines.vert";
      fshaderfn = "./OpenGL/Shaders/lines.frag";
    }
    assert(parse_file_into_str( vshaderfn.c_str(), vertex_shader, 1024 * 256));
    assert(parse_file_into_str( fshaderfn.c_str(), fragment_shader, 1024 * 256));
	
    print_error("-1");
    GLuint vs = glCreateShader (GL_VERTEX_SHADER);
    const GLchar* p = (const GLchar*)vertex_shader;
    glShaderSource(vs, 1, &p, NULL);
    glCompileShader(vs);
	
    // check for compile errors
    int params = -1;
    glGetShaderiv (vs, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) {
      fprintf (stderr, "ERROR: GL shader index %i did not compile\n", vs);
      print_shader_info_log (vs);
      return;
    }
	
    print_error("0");

    GLuint fs = glCreateShader (GL_FRAGMENT_SHADER);
    p = (const GLchar*)fragment_shader;
    glShaderSource (fs, 1, &p, NULL);
    glCompileShader (fs);
	
    // check for compile errors
    glGetShaderiv (fs, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) {
      fprintf (stderr, "ERROR: GL shader index %i did not compile\n", fs);
      print_shader_info_log (fs);
      return;
    }
	
    print_error("1");

    //------------------------------------------------------------
    // Set up program
    //------------------------------------------------------------
    program = glCreateProgram ();
    std::cout << "program " << program << std::endl;
    glAttachShader(program, fs);
    glAttachShader(program, vs);
    glLinkProgram(program);
	
    glGetProgramiv(program, GL_LINK_STATUS, &params);
    if (GL_TRUE != params) {
      fprintf (
          stderr,
          "ERROR: could not link shader programme GL index %i\n",
          program
               );
      print_programme_info_log (program);
      return;
    }
	
    print_error("2");

    vertexId = glGetAttribLocation(program, "vPosition");
    matrixId = glGetUniformLocation (program, "matrix");
    colorId = glGetUniformLocation (program, "color");
    pointSizeId = glGetUniformLocation (program, "pointSize");
    glUseProgram(program);
    print_error("p1");

    matrix = glm::mat4(1.0);
    glUniformMatrix4fv(matrixId, 1, GL_FALSE, glm::value_ptr(matrix));
  }

  GLint getVertexLoc() const {
    return vertexId;
  }

  void useProgram() {
    glUseProgram(program);
    print_error("p");
  }

  void multMatrix(const glm::mat4& m) {
    matrix = matrix * m;
    glUniformMatrix4fv(matrixId, 1, GL_FALSE, glm::value_ptr(matrix));
  }

  void setMatrix(const glm::mat4& m) {
    glUniformMatrix4fv(matrixId, 1, GL_FALSE, glm::value_ptr(m));
  }

  void setColor(const int seed) {
    float_3 color = Color::randomColor(seed);
    glUniform3fv(colorId, 1, color.s);
  }

  void setColor(const float_3& color) {
    glUniform3fv(colorId, 1, color.s);
  }

  void setPointSize(const float& pointSize) {
    glUniform1f(pointSizeId, pointSize);
  }
};

#endif
