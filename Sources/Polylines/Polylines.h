#pragma once

#include <vector>
#include <iostream>
#include <fstream>

#include "./glm/gtc/matrix_transform.hpp"
#include "Shaders/Shaders.hpp"
#include "Quantize/Quantize.h"
extern "C" {
#include "Line/Line.h"
}
#include "clfw.hpp"
#include "Vector/vec.h"
#include "Options/Options.h"
#include "Color/Color.h"
#include <cstring>
#include <iostream>

class PolyLines {
	private:
		static int count;
		int uid;
		int currentColor = 0;
		void readMesh(const string& filename);
	public:
		std::vector<cl_int> lasts;
		std::vector<float2> points;
		std::vector<cl_int> pointColors;
		std::vector<Line> lines;
		cl::Buffer pointsBuffer;
		cl::Buffer linesBuffer;
		cl::Buffer pointColorsBuffer;

		void newLine(const float2& p);
		void addPoint(const float2& p);
		void addLine(const std::vector<float2> newPoints);
		void undoLine();
		void clear();
		void writeToFile(const string &foldername);
		PolyLines(std::vector<std::string> filenames);
		~PolyLines();
};