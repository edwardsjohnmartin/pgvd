#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#define GL_LOG_FILE "gl.log"

#include "gl_utils.h"

#include "../opencl/vec.h"
#include "./Polylines.h"
#include "./LinesProgram.h"
#include "./Octree2.h"
#include "../Resln.h"

// keep track of window size for things like the viewport and the mouse cursor
int g_gl_width = 500;
int g_gl_height = 400;
GLFWwindow* g_window = NULL;

Polylines* lines;
LinesProgram* program;
Octree2* octree;

bool mouseDown = false;
float2 curMouse;

GLFWcursor* arrowCursor;
GLFWcursor* zoomCursor;
bool zoomMode = false;

//------------------------------------------------------------
// refresh
//------------------------------------------------------------
void refresh() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (options.showOctree) {
    octree->render(program);
  }
  lines->render(program);
  glfwSwapBuffers(g_window);
}

void rebuild() {
  octree->build(*lines);
  refresh();
}

//------------------------------------------------------------
// key_callback
//------------------------------------------------------------
void onKey(GLFWwindow* window, int key, int scancode,
           int action, int mods) {
  using namespace std;

  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_C:
        lines->clear();
        octree->build(*lines);
        break;
      case GLFW_KEY_P:
        options.showObjectVertices = !options.showObjectVertices;
        break;
      case GLFW_KEY_O:
        options.showOctree = !options.showOctree;
        break;
      case GLFW_KEY_Z:
        zoomMode = !zoomMode;
        if (zoomMode) {
          glfwSetCursor(g_window, zoomCursor);
        } else {
          glfwSetCursor(g_window, arrowCursor);
        }
        break;
      case GLFW_KEY_Q:
        glfwSetWindowShouldClose(g_window, 1);
        break;
    }
  }

  refresh();
}

void onMouse(GLFWwindow* window, int button, int action, int mods) {
  using namespace std;

  if (action == GLFW_PRESS) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      mouseDown = true;
      lines->newLine(curMouse);
      rebuild();
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    }
  } else if (action == GLFW_RELEASE) {
      mouseDown = false;
  }
}

void onMouseMove(GLFWwindow* window, double xpos, double ypos) {
  using namespace std;

  const float x = (xpos / g_gl_width) * 2 - 1;
  const float y = (ypos / g_gl_height) * 2 - 1;
  curMouse = make_float2(x, -y);

  if (mouseDown) {
    lines->addPoint(curMouse);
    rebuild();
  }
}

int main_new(int argc, char** argv) {
  using namespace std;

  restart_gl_log();
  start_gl();
  print_error("new a");
  glfwSetWindowTitle(g_window, "Parallel GVD");

  glfwSetKeyCallback(g_window, onKey);
  glfwSetMouseButtonCallback(g_window, onMouse);
  glfwSetCursorPosCallback(g_window, onMouseMove);

  GLFWcursor* arrowCursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
  GLFWcursor* zoomCursor = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
  glfwSetCursor(g_window, arrowCursor);
  // glfwSetCursor(g_window, zoomCursor);

  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear (GL_COLOR_BUFFER_BIT);

  octree = new Octree2();
  octree->processArgs(argc, argv);
  lines = new Polylines();

  program = new LinesProgram();
	
  refresh();

  while (!glfwWindowShouldClose(g_window)) {
    // Refresh here for animation
    // refresh();
    
    glfwPollEvents ();
  }
	
  glfwTerminate();
  return 0;
}

int main_old() {
  restart_gl_log();
  start_gl();
  print_error("old a");

  glfwSetKeyCallback(g_window, onKey);

  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear (GL_COLOR_BUFFER_BIT);

  //------------------------------------------------------------
  // Initialize buffer arrays
  //------------------------------------------------------------
  static const int n = 100;
  float3 points[n];
  float3 vels[n];
  for (int i = 0; i < n; ++i) {
    points[i][0] = 2.0*rand()/static_cast<float>(RAND_MAX)-1.0;
    points[i][1] = 2.0*rand()/static_cast<float>(RAND_MAX)-1.0;
    points[i][2] = 2.0*rand()/static_cast<float>(RAND_MAX)-1.0;
    // vels[i] = float3(0);
  }

  // cout << bb(points, n) << endl;

  GLuint points_vbo;
  glGenBuffers (1, &points_vbo);
  glBindBuffer (GL_ARRAY_BUFFER, points_vbo);
  glBufferData (GL_ARRAY_BUFFER, n*sizeof(float3), points, GL_STATIC_DRAW);
	
  GLuint vao;
  glGenVertexArrays (1, &vao);
  glBindVertexArray (vao);
  glBindBuffer (GL_ARRAY_BUFFER, points_vbo);
  glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray (0);
	
  //------------------------------------------------------------
  // Initialize shaders
  //------------------------------------------------------------
  char vertex_shader[1024 * 256];
  char fragment_shader[1024 * 256];
  assert (parse_file_into_str ("../viewer/shaders/test.vert", vertex_shader, 1024 * 256));
  assert (parse_file_into_str ("../viewer/shaders/test.frag", fragment_shader, 1024 * 256));
	
  GLuint vs = glCreateShader (GL_VERTEX_SHADER);
  const GLchar* p = (const GLchar*)vertex_shader;
  glShaderSource (vs, 1, &p, NULL);
  glCompileShader (vs);
	
  // check for compile errors
  int params = -1;
  glGetShaderiv (vs, GL_COMPILE_STATUS, &params);
  if (GL_TRUE != params) {
    fprintf (stderr, "ERROR: GL shader index %i did not compile\n", vs);
    print_shader_info_log (vs);
    return 1; // or exit or something
  }
	
  GLuint fs = glCreateShader (GL_FRAGMENT_SHADER);
  p = (const GLchar*)fragment_shader;
  glShaderSource (fs, 1, &p, NULL);
  glCompileShader (fs);
	
  // check for compile errors
  glGetShaderiv (fs, GL_COMPILE_STATUS, &params);
  if (GL_TRUE != params) {
    fprintf (stderr, "ERROR: GL shader index %i did not compile\n", fs);
    print_shader_info_log (fs);
    return 1; // or exit or something
  }
	
  //------------------------------------------------------------
  // Set up program
  //------------------------------------------------------------
  GLuint program = glCreateProgram ();
  std::cout << "program " << program << std::endl;
  glAttachShader (program, fs);
  glAttachShader (program, vs);
  glLinkProgram (program);
	
  glGetProgramiv (program, GL_LINK_STATUS, &params);
  if (GL_TRUE != params) {
    fprintf (
        stderr,
        "ERROR: could not link shader programme GL index %i\n",
        program
             );
    print_programme_info_log (program);
    return false;
  }
	
  GLfloat matrix[] = {
    1.0f, 0.0f, 0.0f, 0.0f, // first column
    0.0f, 1.0f, 0.0f, 0.0f, // second column
    0.0f, 0.0f, 1.0f, 0.0f, // third column
    0.0f, 0.0f, 0.0f, 1.0f // fourth column
  };
	
  int matrix_location = glGetUniformLocation (program, "matrix");
  int center_location = glGetUniformLocation (program, "center");
  int t_location = glGetUniformLocation (program, "t");
  glUseProgram (program);
  print_error("old b");
  glUniformMatrix4fv(matrix_location, 1, GL_FALSE, matrix);
  float center[] = { 0.0, 0.0, 0.0 };
  glUniform3fv(center_location, 1, center);
  float t = 0;
  glUniform1f(t_location, t);
	
  // glEnable (GL_CULL_FACE); // cull face
  // glCullFace (GL_BACK); // cull back face
  // glFrontFace (GL_CW); // GL_CCW for counter clock-wise
	
  float speed = 1.0f; // move at 1 unit per second
  float last_position = 0.0f;
  while (!glfwWindowShouldClose(g_window)) {
    // add a timer for doing animation
    static double previous_seconds = glfwGetTime ();
    double current_seconds = glfwGetTime ();
    // double elapsed_seconds = current_seconds - previous_seconds;
    previous_seconds = current_seconds;
		
    _update_fps_counter (g_window);
    // wipe the drawing surface clear
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glViewport (0, 0, g_gl_width, g_gl_height);
		
    //
    // Note: this call is not necessary, but I like to do it anyway before any
    // time that I call glDrawArrays() so I never use the wrong shader programme
    glUseProgram (program);
		
    // update the matrix
    // - you could simplify this by just using sin(current_seconds)
    // matrix[12] = elapsed_seconds * speed + last_position;
    last_position = matrix[12];
    if (fabs (last_position) > 1.0) {
      speed = -speed;
    }
    //
    // Note: this call is related to the most recently 'used' shader programme
    glUniformMatrix4fv (matrix_location, 1, GL_FALSE, matrix);
		
    // Update t
    t = fmin(1.0, t + 0.01);
    glUniform1f(t_location, t);

    // Note: this call is not necessary, but I like to do it anyway before any
    // time that I call glDrawArrays() so I never use the wrong vertex data
    glBindVertexArray (vao);
    // draw points 0-3 from the currently bound VAO with current in-use shader
    // glDrawArrays (GL_TRIANGLES, 0, 3);
    // glDrawArrays (GL_LINE_LOOP, 0, 3);

    // float3 delta1[n];
    // float3 delta2[n];
    // float3 delta3[n];
    // for (int i = 0; i < n; ++i) {
    //   delta1[i] = rule1(points, n, i, o);
    //   delta2[i] = rule2(points, n, i, o);
    //   delta3[i] = rule3(points, vels, n, i, o);
    // }
    // for (int i = 0; i < n; ++i) {
    //   // call avoidBoundaries
    //   // b->vel = b->vel.rotate(angle);
    //   // b->vel = b->vel + newPos1 + newPos2 + newVel3;
    //   // b->pos = b->pos + b->vel;

    //   // glm::mat4 angle = avoidBoundaries(points, vels, n, i);
    //   // // Rotate by angle
    //   // float3 result(0);
    //   // float x = vels[i][0];
    //   // float y = vels[i][1];
    //   // float z = vels[i][2];
    //   // result[0] = x * cos(angle) - y * sin(angle);
    //   // result[1] = x * sin(angle) + y * cos(angle);
    //   // result[2] = z;

    //   // vels[i] = result;
    //   // Manual avoid boundaries
    //   for (int j = 0; j < 3; ++j) {
    //     if (vels[i][j] > 0.0 && points[i][j] > 0.9) {
    //       vels[i][j] *= -1;
    //     } else if (vels[i][j] < 0.0 && points[i][j] < -0.9) {
    //       vels[i][j] *= -1;
    //     }
    //   }

    //   vels[i] = vels[i] + delta1[i] + delta2[i] + delta3[i];
    //   points[i] = points[i] + vels[i];
    // }

    glBindBuffer (GL_ARRAY_BUFFER, points_vbo);
    glBufferData (GL_ARRAY_BUFFER, n*sizeof(float3), points, GL_STATIC_DRAW);

    print_error("ghi");
    // glEnable(GL_POINT_SPRITE);
    print_error("def");
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    print_error("abc");
    glDrawArrays (GL_POINTS, 0, n);
    
    // update other events like input handling 
    glfwPollEvents ();
    // handleKeyPress();
    // put the stuff we've been drawing onto the display
    glfwSwapBuffers (g_window);
  }
	
  // close GL context and any other GLFW resources
  glfwTerminate();
  return 0;
}

int main(int argc, char** argv) {
  const bool old = false;
  if (old)
    main_old();
  else
    main_new(argc, argv);
}

