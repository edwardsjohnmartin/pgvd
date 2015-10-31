#ifndef __LINE_PROGRAM_H__
#define __LINE_PROGRAM_H__

#include <assert.h>

#include "./glm/mat4x4.hpp"
#include "./glm/gtc/type_ptr.hpp"

#include "gl_utils.h"

#include "./Color.h"

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
    assert(parse_file_into_str(
        "../viewer/shaders/lines-v1.2.vert", vertex_shader, 1024 * 256));
    assert(parse_file_into_str(
        "../viewer/shaders/lines.frag", fragment_shader, 1024 * 256));
	
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

  void setColor(const int seed) {
    float3 color = Color::randomColor(seed, make_float3(1, 0, 0));
    glUniform3fv(colorId, 1, color.s);
  }

  void setColor(const float3& color) {
    glUniform3fv(colorId, 1, color.s);
  }

  void setPointSize(const float& pointSize) {
    glUniform1f(pointSizeId, pointSize);
  }
};

#endif