uniform vec3 color;

// in vec3 color;
//out vec4 frag_color;

void main() {
  gl_FragColor = vec4(color, 1.0);
}
