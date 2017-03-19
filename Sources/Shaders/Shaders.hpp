#pragma once
#include "./Programs/ShaderProgram.hpp"
namespace Shaders {
	extern ShaderProgram* boxProgram;
	extern ShaderProgram* planeProgram;
  extern ShaderProgram* pointProgram;
  extern ShaderProgram* lineProgram;
  //extern ShaderProgram* sketchBoxProgram;
  void create();
}
