#include "Shaders.hpp"
namespace Shaders {
  ShaderProgram* boxProgram;
  ShaderProgram* lineProgram;
  ShaderProgram* sketchProgram;
  ShaderProgram* sketchLineProgram;
  //ShaderProgram* sketchBoxProgram;
  void create() {
    boxProgram = new ShaderProgram("./SharedSources/OpenGL/Shaders/boxes.vert", "./SharedSources/OpenGL/Shaders/boxes.frag");
    lineProgram = new ShaderProgram("./SharedSources/OpenGL/Shaders/lines.vert", "./SharedSources/OpenGL/Shaders/lines.frag");
    sketchProgram = new ShaderProgram("./SharedSources/OpenGL/Shaders/sketchPoint.vert", "./SharedSources/OpenGL/Shaders/sketchPoint.frag");
    sketchLineProgram = new ShaderProgram("./SharedSources/OpenGL/Shaders/sketchLine.vert", "./SharedSources/OpenGL/Shaders/sketchLine.frag");
    //sketchBoxProgram = new ShaderProgram("./SharedSources/OpenGL/Shaders/sketchBox.vert", "./SharedSources/OpenGL/Shaders/sketchBox.frag");
  }
}
