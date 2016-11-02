#version 410

layout(location = 0) in vec4 position;

uniform float pointSize;

void main() {
  gl_PointSize = pointSize+5.0;
  gl_Position = vec4(position.x, position.y, position.z, 1.0);
}
