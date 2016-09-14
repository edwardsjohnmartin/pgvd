#include "Shaders.hpp"
namespace Shaders {
  ShaderProgram* boxProgram;
  ShaderProgram* lineProgram;
  void create() {
    boxProgram = new ShaderProgram("./opengl/shaders/boxes.vert", "./opengl/shaders/boxes.frag");
    lineProgram = new ShaderProgram("./opengl/shaders/lines.vert", "./opengl/shaders/lines.frag");
  }
}