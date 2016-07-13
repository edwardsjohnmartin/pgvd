#include "Shaders.hpp"
namespace Shaders {
  ShaderProgram* boxProgram;
  void create() {
    boxProgram = new ShaderProgram("./opengl/shaders/boxes.vert", "./opengl/shaders/boxes.frag");
  }
}