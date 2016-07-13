#version 410

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 offset;
layout(location = 2) in vec3 scale;
layout(location = 3) in vec3 color;

uniform mat4 matrix;
uniform float point_size;

out vec4 fColor;

void main() {
  gl_PointSize = point_size;
  gl_Position = matrix * vec4(position + offset, 1.0) * scale;
  fColor = vec4(color, 1.0);
}
