#include "Shaders.hpp"
namespace Shaders {
	ShaderProgram* planeProgram;
	ShaderProgram* boxProgram;
  ShaderProgram* pointProgram;
  ShaderProgram* lineProgram;
  //ShaderProgram* sketchBoxProgram;
  void create() {
		planeProgram = new ShaderProgram("./Sources/OpenGL/Shaders/plane.vert", "./Sources/OpenGL/Shaders/plane.frag");
		boxProgram = new ShaderProgram("./Sources/OpenGL/Shaders/boxes.vert", "./Sources/OpenGL/Shaders/boxes.frag");
		lineProgram = new ShaderProgram("./Sources/OpenGL/Shaders/lines.vert", "./Sources/OpenGL/Shaders/lines.frag");
  	pointProgram = new ShaderProgram("./Sources/OpenGL/Shaders/points.vert", "./Sources/OpenGL/Shaders/points.frag");
    //sketchBoxProgram = new ShaderProgram("./Sources/OpenGL/Shaders/sketchBox.vert", "./Sources/OpenGL/Shaders/sketchBox.frag");
  }
}
