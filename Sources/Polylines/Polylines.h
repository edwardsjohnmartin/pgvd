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
#include "Vector/vec.h"
#include "Options/Options.h"
#include "Color/Color.h"
#include <cstring>
#include <iostream>

class PolyLines {
private:
	std::vector<floatn> points;
	// Index of last+1 point in a line
	std::vector<int> lasts;
	std::vector<float3> colors;

	GLuint pointsVboId;
	GLuint pointsVaoId;

public:
	PolyLines(std::vector<std::string> filenames) {
		/* Read files */
		for (int i = 0; i < Options::filenames.size(); ++i)
			readMesh(Options::filenames[i]);
		srand(time(NULL));
	}

	~PolyLines() {
	}

	void clear() {
		points.clear();
		lasts.clear();
		colors.clear();
	}

	void addPoint(const floatn& p) {
		if (lasts.size() == 0) return;
		using namespace std;

		points.push_back(p);

		lasts.back() = points.size();
	}

	void newLine(const floatn& p) {
		lasts.push_back(0);
		colors.push_back(Color::randomColor());
		addPoint(p);
	}

	void undoLine() {
		if (lasts.size() == 0) return;
		int numElements = lasts[lasts.size() - 1] - ((lasts.size() == 1) ? 0 : lasts[lasts.size() - 2]);
		points.resize(points.size() - numElements);
		lasts.pop_back();
		colors.pop_back();
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
			newLine({ x, y });
		}

		while (in >> x && in >> y) {
			addPoint({ x, y });
		}

		in.close();
	}

	void writeToFile(const string &foldername) {
		auto polygons = getPolygons();
		ofstream out;
		ofstream subout;
		out.open(foldername + "/" + "files" + ".txt");
		out.clear();
		for (int i = 0; i < polygons.size(); ++i) {
			cout << "writing to " << foldername << "/" << i << ".bin" << endl;
			out << foldername << "/" << i << ".bin" << endl;
			subout.open(foldername + "/" + std::to_string(i) + ".bin");
			subout.clear();
			auto polygon = polygons[i];
			for (int j = 0; j < polygon.size(); j++) {
				subout << polygon[j].x << " " << polygon[j].y << endl;
			}
			subout.close();
		}
		out.close();
	}

	std::vector<std::vector<floatn>> getPolygons() const {
		using namespace std;
		vector<vector<floatn>> ret;

		int first = 0;
		for (int i = 0; i < lasts.size(); ++i) {
			const int last = lasts[i];
			vector<floatn> polygon;
			for (int j = first; j < last; ++j) {
				polygon.push_back(points[j]);
			}
			if (polygon.size() > 1) {
				ret.push_back(polygon);
			}
			first = lasts[i];
		}
		return ret;
	}

	std::vector<std::vector<floatn>> getQuantizedPolygons(const floatn minimum, const int reslnWidth, const float bbWidth) const {
		using namespace std;
		vector<vector<floatn>> ret;

		int first = 0;
		for (int i = 0; i < lasts.size(); ++i) {
			const int last = lasts[i];
			vector<floatn> polygon;
			for (int j = first; j < last; ++j) {
				intn quantized = QuantizePoint(&points[j], &minimum, reslnWidth, bbWidth);
				floatn unquantized = UnquantizePoint(&quantized, &minimum, reslnWidth, bbWidth);
				polygon.push_back(unquantized);
			}
			if (polygon.size() > 1) {
				ret.push_back(polygon);
			}
			first = lasts[i];
		}
		return ret;
	}

	std::vector<Line> getLines() const {
		using namespace std;
		vector<Line> lines;
		int totalSkips = 0; //we skip single point polylines.
		int first = 0;
		//For each polyline 
		for (int i = 0; i < lasts.size(); ++i) {
			int last = lasts[i];
			first = (i != 0) ? lasts[i - 1] : i;
			//if the polyline has more than one point
			if (last - first > 1) {
				//then add the lines in this polyline
				for (int j = 0; j < (last - first) - 1; ++j) {
					Line line;
					line.first = first + j - totalSkips;
					line.second = first + j + 1 - totalSkips;
					line.color = i;
					lines.push_back(line);
				}
			}
			else totalSkips++;
		}
		return lines;
	}
};
