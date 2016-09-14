#version 410

layout(location = 0) in vec3 position;

uniform float pointSize;

void main() {
  gl_PointSize = pointSize+5.0;
  gl_Position = vec4(position, 1.0);
}
