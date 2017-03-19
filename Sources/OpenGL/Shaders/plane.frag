#version 410
uniform sampler2D texture;

in vec2 ftexcoord;
out vec4 finalColor;

void main() {
  finalColor = texture2D(texture, ftexcoord * .5 + .5);
}
