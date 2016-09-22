#include "Shaders.hpp"
namespace Shaders {
  ShaderProgram* boxProgram;
  ShaderProgram* lineProgram;
  void create() {
    boxProgram = new ShaderProgram("./OpenGL/Shaders/boxes.vert", "./OpenGL/Shaders/boxes.frag");
    lineProgram = new ShaderProgram("./OpenGL/Shaders/lines.vert", "./OpenGL/Shaders/lines.frag");
  }
}
