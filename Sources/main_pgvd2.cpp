#include <iostream>
#include "GLUtilities/gl_utils.h"
#include "clfw.hpp"
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
void InitializeGLFW(int width = 1024, int height = 1024) {
  using namespace GLUtilities;
  GLUtilities::window_height = height;
  GLUtilities::window_width = width;
  restart_gl_log();
  start_gl();
  print_gl_error();
  glfwSetWindowTitle(window, "Parallel GVD");
  InitializeGLFWEventCallbacks();
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  glfwSwapBuffers(window);
}

int processArgs(int argc, char** argv) {
  using namespace Options;

  int i = 1;
  // if (argc > 1) {
  bool stop = false;
  while (i < argc && !stop) {
    stop = true;
    if (processArg(i, argv)) {
      stop = false;
    }
  }
  // if (o.help) {
  //   PrintUsage();
  //   exit(0);
  // }
  for (; i < argc; ++i) {
    string filename(argv[i]);
    cout << filename << endl;
    // ReadMesh(filename);
    // Data::lines->newLine({ md.x, -md.y });
    // Data::lines->addPoint({ md.x, -md.y });
    // Data::octree->build(Data::lines);
  }

  // int num_edges = 0;
  // for (int i = 0; i < polygons.size(); ++i) {
  //   num_edges += polygons[i].size();
  // }

  // PrintCommands();
  // cout << "Number of objects: " << polygons.size() << endl;
  // cout << "Number of polygon edges: " << num_edges << endl;
}

int main(int argc, char** argv) {
  using namespace std;
  CLFW::Initialize(true, true, 2);
  InitializeGLFW(512, 512);
  Shaders::create();
  Data::lines = new PolyLines();
	//Data::lines->newLine({ -.5,-.5 });
	//Data::lines->addPoint({ .5,.5 });

  Options::showObjects = true;
  Options::showOctree = true;
  Options::max_level = 6;
  Options::device = -1;

  Data::octree = new Octree2();

  processArgs(argc, argv);

  /* Event loop */
  while (!glfwWindowShouldClose(GLUtilities::window)) {
    glfwPollEvents();
    refresh();
  }
  glfwTerminate();
  return 0;
}
