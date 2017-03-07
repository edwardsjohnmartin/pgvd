#pragma once
#include "GLUtilities/gl_utils.h"
#include "Options/Options.h"
#include "GlobalData/data.h"
#include "Shaders/Shaders.hpp"

void key_cb(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_cb(GLFWwindow* window, int button, int action, int mods);
void mouse_move_cb(GLFWwindow* window, double xpos, double ypos);
void scroll_cb(GLFWwindow* window, double xoffset, double yoffset);
void resize_cb(GLFWwindow* window, int width, int height);
void focus_cb(GLFWwindow* window, int focused);
void refresh();