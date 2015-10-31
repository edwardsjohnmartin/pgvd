TODO pgvd2
* Clear lines when 'c' is pressed
* Fix bug on intersecting polylines
* Convert Karras algorithm to GPU implementation
* Add support for passing in object filenames at the command-line (code for
  this can be found in the old gvd project)
* Implement zoom
  * Figure out why cursor isn't changing on pressing 'z'
* Implement push/pop matrix in LinesProgram


To build with CMake
------------------------------------------------------------
* On Linux/Mac
  1. mkdir release
  2. cd release
  3. cmake ..
  4. make
* On Windows
  1. Open the CMake GUI.
  2. Select the pgvd dir for the source
  3. Select a prefered build location INSIDE the pgvd directory. 
    * Since kernels and shaders are relative to the pgvd dir, placing the build folder in ./pgvd is required.
  4. Click Configure
  5. Select visual studio 12 2013 
  6. Click Generate
  7. Follow the Visual Studio Setup

To Setup Visual Studio
------------------------------------------------------------
1. Open the PGVD.sln in your build folder.
2. Right click pgvd2 in the solution explorer and click properties.
3. In Configuration Properties -> C/C++ -> General -> Additional Include Directories
  * Add "<git directory>\pgvd\dependencies\glew-1.13.0\include"
  * Add "<git directory>\pgvd\dependencies\glfw-3.1.2.bin.WIN32\include"
4. In Configuration Properties -> Linker -> General -> Additional Library Directories
  * Add "<git directory>\pgvd\dependencies\glew-1.13.0\lib\Release\Win32"
  * Add "<git directory>\pgvd\dependencies\glfw-3.1.2.bin.WIN32\lib-vc2013"
5. In Configuration Properties -> Linker -> Input -> Additional Dependencies
  * Add "glfw3.lib"
  * Add "glew32.lib"
  * Add "opengl32.lib"
6. Click Ok
7. Click "Solution 'PGVD'" in the solution explorer.
  * Set Startup Project to "pgvd2"
8. Open "<git directory>\pgvd\dependencies\glew-1.13.0\bin\Release\Win32
  * Copy glew32.dll to <your build folder>\Debug\



Style guide
These are code style guidelines for this project. Not all
code may conform yet -- it is a work in progress.
------------------------------------------------------------
* Namespaces, structs and classes are named with upper CamelCase.
* Data members, variables and functions are named with lower camelCase.
* Classes and structs declare data members at the beginning of the class.
* Global variables and functions that are declared in <filename>.h are
  placed in the namespace <filename>.
* OpenGL identifiers end with "Id".
* Options that are modified either from the command-line or using keystrokes
  are placed in options.h/cpp. A global variable options references these.
