attribute vec2 vertex_position;

uniform mat4 matrix;
// uniform vec3 center;
uniform vec3 color;

// varying vec3 color;

void main() {
  // vec3 v = center - vertex_position;
  
  // color = vec3(0.0, 0.0, 0.0);//vertex_colour;
  gl_PointSize = 5.0;
  gl_Position = matrix * vec4(vertex_position, 0.0, 1.0);
}
