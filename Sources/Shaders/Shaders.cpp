#include "Shaders.hpp"
namespace Shaders {
  ShaderProgram* boxProgram;
  ShaderProgram* lineProgram;
  ShaderProgram* sketchProgram;
  ShaderProgram* sketchLineProgram;
  //ShaderProgram* sketchBoxProgram;
  void create() {
    boxProgram = new ShaderProgram("./Sources/OpenGL/Shaders/boxes.vert", "./Sources/OpenGL/Shaders/boxes.frag");
    lineProgram = new ShaderProgram("./Sources/OpenGL/Shaders/lines.vert", "./Sources/OpenGL/Shaders/lines.frag");
    sketchProgram = new ShaderProgram("./Sources/OpenGL/Shaders/sketchPoint.vert", "./Sources/OpenGL/Shaders/sketchPoint.frag");
    sketchLineProgram = new ShaderProgram("./Sources/OpenGL/Shaders/sketchLine.vert", "./Sources/OpenGL/Shaders/sketchLine.frag");
    //sketchBoxProgram = new ShaderProgram("./Sources/OpenGL/Shaders/sketchBox.vert", "./Sources/OpenGL/Shaders/sketchBox.frag");
  }
}
