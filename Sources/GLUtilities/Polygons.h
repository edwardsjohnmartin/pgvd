#pragma once
#include "clfw.hpp"
#include "glm/glm.hpp"
#include "Line/Line.h"
#include "OctreeDefinitions/defs.h"
#include "Vector/vec.h"
#include <fstream>
class Polygons {
	private:
		static int count;
		int uid;
		void readMesh(const std::string& filename);
	public:
		std::vector<float2> points;
		std::vector<Line> lines;
		std::vector<cl_int> pointColors;
		std::vector<cl_int> numPointsInPolygon;
		std::vector<cl_int> lasts;
		cl::Buffer pointsBuffer;
		cl::Buffer linesBuffer;
		cl::Buffer pointColorsBuffer;
		void addPolygon(std::vector<float2> points);
		void removePolygon(int index);
		void changePolygonColor(int index, float4 color);
		void movePolygon(int index, glm::mat4 matrix);
		void clear();
		Polygons(std::vector<std::vector<float2>> polygons);
		Polygons(std::vector<std::string> filenames);
		~Polygons();
};