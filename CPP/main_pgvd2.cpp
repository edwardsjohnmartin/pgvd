#include <iostream>
#include "clfw.hpp"
#include "GLUtilities/gl_utils.h"
#include "Events/events.h"
#include "Shaders/Shaders.hpp"
#include "Polylines/Polylines.h"
#include "GlobalData/data.h"
#include "Options/Options.h"

void InitializeGLFWEventCallbacks() {
  using namespace GLUtilities;
  glfwSetKeyCallback(window, key_cb);
  glfwSetMouseButtonCallback(window, mouse_cb);
  glfwSetCursorPosCallback(window, mouse_move_cb);
  glfwSetWindowSizeCallback(window, resize_cb);
  glfwSetWindowFocusCallback(window, focus_cb);
}
void InitializeGLFW(int width = 512, int height = 512) {
  using namespace GLUtilities;
  GLUtilities::window_height = height;
  GLUtilities::window_width = width;
  restart_gl_log();
  start_gl();
  print_error("new a");
  glfwSetWindowTitle(window, "Parallel GVD");
  InitializeGLFWEventCallbacks();
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  glfwSwapBuffers(window);
}

int main(int argc, char** argv) {
  using namespace std;
  CLFW::Initialize(false, true, 1);
  InitializeGLFW();
  Shaders::create();
  Data::lines = new PolyLines();
  Data::lines->newLine({ -.5,-.5 });
  Data::lines->addPoint({ .5,.5 });
  Data::lines->addPoint({ .0,.0 });

  Data::octree = new Octree2();

  Options::showObjects = true;
  Options::showOctree = true;
  /* Event loop */
  while (!glfwWindowShouldClose(GLUtilities::window)) {
    glfwPollEvents();
    refresh();
  }
  glfwTerminate();
  return 0;
}

////
////int main(int argc, char** argv) {
////  srand(static_cast <unsigned> (time(0)));
////  using namespace std;
////
////  checkError(CLFW::Initialize(false, true, 1));
////
////  InitializeOpenGL();
////  Shaders::create();
////
////  octree = new Octree2();
////  octree->processArgs(argc, argv);
////  lines = new PolyLines();
////  lines->newLine({ 0.0f, 0.0f });
////  lines->addPoint({ 0.1f, 0.1f});
////  //point1 = point2 = make_float2(0.0f, 0.0f);
////
////  program = new LinesProgram();
////
////  //for testing
////  octree->program = program;
////
////  //glm::mat4 matrix = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f);
////  //program->setMatrix(matrix);
////	
////  /*for (const string& f : options.filenames) {
////    ifstream in(f.c_str());
////    bool first = true;
////    while (!in.eof()) {
////      double x, y;
////      in >> x >> y;
////      if (options.jitter) {
////        double s = 100;
////        x += rand() / (RAND_MAX*s) - 1/(s*2);
////        y += rand() / (RAND_MAX*s) - 1/(s*2);
////      }
////      if (!in.eof()) {
////        if (first) {
////          lines->newLine(make_float2(x, y));
////          first = false;
////        } else {
////          lines->addPoint(make_float2(x, y));
////        }
////      }
////    }
////    in.close();
////  }
////  rebuild();
////
////  refresh();
////
////*/
////  //float radius = 1.0;
////  ////500000
////  //for (int i = 0; i < 3000; i++) {
////  //  radius -= .0005;
////  //  addpt(radius);
////  //  rebuild();
////  //}
////  while (!glfwWindowShouldClose(g_window)) {
////    glfwPollEvents();
////  }
////  glfwTerminate();
////  return 0;
////}
