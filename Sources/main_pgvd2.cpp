#include <iostream>
#include "GLUtilities/gl_utils.h"
#include "clfw.hpp"
#include "Events/events.h"
#include "Shaders/Shaders.hpp"
#include "Polylines/Polylines.h"
#include "GlobalData/data.h"
#include "Options/Options.h"

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

void InitializeGLFWEventCallbacks() {
  using namespace GLUtilities;
  glfwSetKeyCallback(window, key_cb);
  glfwSetMouseButtonCallback(window, mouse_cb);
  glfwSetCursorPosCallback(window, mouse_move_cb);
  glfwSetWindowSizeCallback(window, resize_cb);
  glfwSetWindowFocusCallback(window, focus_cb);
}

void fixResolution(int& width, int& height) {
    const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    width = min(width, mode->width);
    height = min(height, mode->height);
    if (width < height) {
      height = width;
    } else {
      width = height;
    }
}

void InitializeGLFW(int width = 1024, int height = 1024) {
  using namespace GLUtilities;
  GLUtilities::window_height = height;
  GLUtilities::window_width = width;
  restart_gl_log();
  start_gl();
  fixResolution(width, height);
  GLUtilities::window_height = height;
  GLUtilities::window_width = width;

  print_gl_error();
  glfwSetWindowTitle(window, "Parallel GVD");
  InitializeGLFWEventCallbacks();
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  glfwSwapBuffers(window);
  glfwSwapInterval(1);
}

void readMesh(const string& filename) {
  ifstream in(filename.c_str());
  if (!in) {
    cerr << "Failed to read " << filename << endl;
    return;
  }
  float x, y;
  in >> x >> y;
  if (!in.eof()) {
    Data::lines->newLine({ x, y });
  }

  while (!in.eof()) {
    in >> x >> y;
    if (!in.eof()) {
      Data::lines->addPoint({ x, y });
    }
  }
  in.close();
}

int processArgs(int argc, char** argv) {
  using namespace Options;

  static log4cplus::Logger logger =
      log4cplus::Logger::getInstance("processArgs");

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
    LOG4CPLUS_DEBUG(logger, "Read " << filename);
    filenames.push_back(filename);
  }
  //TODO: fix this.
  return 0;
}

// Tests
// ./QUADTREE -d 1 -l 24 data/maze/poly*.dat
// 
// These two give different results:
//     ./QUADTREE -d 0 -l 24 data/test1-*.dat // works correctly
//     ./QUADTREE -d 1 -l 24 data/test1-*.dat // doesn't resolve conflict cells (although it does find them)
int main(int argc, char** argv) {
  using namespace std;

  // Initialize log4cplus
  log4cplus::initialize();
  ifstream config_in("log4cplus.config");
  if (config_in) {
    config_in.close();
    log4cplus::PropertyConfigurator config("log4cplus.config");
    config.configure();
  } else {
    log4cplus::BasicConfigurator config;
    config.configure();
  }

  processArgs(argc, argv);

  // Options::showObjects = true;
  // Options::showOctree = true;
  // Options::max_level = 6;
  // Options::device = -1;

  CLFW::Initialize(true, true, 2);
  InitializeGLFW();
  Shaders::create();
  Data::lines = new PolyLines();
  Data::octree = new Octree2();

  for (int i = 0; i < Options::filenames.size(); ++i) {
    readMesh(Options::filenames[i]);
  }
  if (!Options::filenames.empty()) {
    Data::octree->build(Data::lines);
    refresh();
  }

  /* Event loop */
  while (!glfwWindowShouldClose(GLUtilities::window)) {
    glfwPollEvents();
    refresh();
  }
  glfwTerminate();
  return 0;
}
