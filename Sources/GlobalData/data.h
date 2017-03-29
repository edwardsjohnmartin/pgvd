#pragma once
#include "Polylines/Polylines.h"
#include "Octree/Octree2.h"
struct Gear {
	float outerRadius;
	float innerRadius;
	int numTeeth;
	float toothThickness;
	float dAngle;
	vector<float2> points;
	glm::mat4 matrix;
};

//R - Number of teeth in ring gear
//S - Number of teeth in sun (middle) gear
//P - Number of teeth in planet gears
//R = 2 * p + s

//Tr - Turns of ring gear.
//Ts - Turns of sun gear.
//Ty - Turns of the planetary gear carrier
//(R + S) * Ty = R * Tr + Ts * s
struct GearInfo {
	int R;
	int S;
	int P;
	float Tr;
	float Ts;
	float Ty;
};

namespace Data {
	extern Polygons *polygons;
	extern PolyLines *lines;
  extern Quadtree *quadtree;
	extern vector<Gear> gears;
	extern GearInfo gearInfo;
};