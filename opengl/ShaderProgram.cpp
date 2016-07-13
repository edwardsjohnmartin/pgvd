#include "ShaderProgram.hpp"

ShaderProgram::ShaderProgram(const string vertShader, const string fragShader) {
  program = create_program_from_files(vertShader.c_str(), fragShader.c_str());
  getAttributes();
  getUniforms();
}

void ShaderProgram::getAttributes() {
  position_id = glGetAttribLocation(program, "position");
  offset_id = glGetAttribLocation(program, "offset");
  scale_id = glGetAttribLocation(program, "scale");
  color_id = glGetAttribLocation(program, "color");
}

void ShaderProgram::getUniforms() {
  matrix_id = glGetUniformLocation(program, "matrix");
  pointSize_id = glGetUniformLocation(program, "point_size");
}

void ShaderProgram::use() {
  glUseProgram(program);
}
