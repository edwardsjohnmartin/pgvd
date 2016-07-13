#version 410

uniform vec3 color;

in vec4 fColor;
out vec4 finalColor;
void main() {
  finalColor = fColor;
}
