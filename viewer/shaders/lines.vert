#version 410

layout(location = 0) in vec3 vPosition;

uniform mat4 matrix;
uniform vec3 color;
uniform float pointSize;

// out vec3 color;

void main() {
  gl_PointSize = pointSize;
  gl_Position = matrix * vec4(vPosition, 1.0);
}
