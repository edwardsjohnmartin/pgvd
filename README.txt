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
mkdir release
cd release
cmake ..
make


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
