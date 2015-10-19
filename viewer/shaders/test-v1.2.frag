// varying vec3 color;
uniform vec3 color;

void main() {
  // gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
  gl_FragColor = vec4(color, 1.0);
}
