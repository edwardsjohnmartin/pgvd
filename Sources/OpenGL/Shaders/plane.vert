#version 410

layout(location = 0) in vec2 position;

uniform vec3 u_offset;
uniform float width;
uniform float height;
uniform mat4 matrix;

out vec2 ftexcoord;

void main() {
	vec3 poswithoffset = vec3(position.x * width, position.y * height, 0.0) + u_offset;
  	gl_Position = matrix * vec4(position, 0.0, 1.0);
  	ftexcoord = vec2(position.x, position.y);
}
