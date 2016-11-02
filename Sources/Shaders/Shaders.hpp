#pragma once
#include "./Programs/ShaderProgram.hpp"
namespace Shaders {
  extern ShaderProgram* boxProgram;
  extern ShaderProgram* lineProgram;
  extern ShaderProgram* sketchProgram;
  extern ShaderProgram* sketchLineProgram;
  //extern ShaderProgram* sketchBoxProgram;
  void create();
}