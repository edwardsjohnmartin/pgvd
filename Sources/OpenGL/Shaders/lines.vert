#version 410

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;

uniform mat4 matrix;
uniform float point_size;

out vec4 fColor;

void main() {
  gl_PointSize = point_size;
  gl_Position = matrix * position;
  fColor = color;
}