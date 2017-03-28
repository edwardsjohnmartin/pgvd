#include "Polygons.h"
#include "Kernels/Kernels.h"

int Polygons::count = 0;

Polygons::Polygons(std::vector<std::vector<float2>> polygons) {
	uid = count;
	count++;

	for (int i = 0; i < polygons.size(); ++i) {
		addPolygon(polygons[i]);
	}
}

Polygons::Polygons(std::vector<std::string> filenames) {
	uid = count;
	count++;

	/* Read each file's points into it's own polygon. */
	for (std::string s : filenames) readMesh(s);
}

Polygons::~Polygons() {
	count--;
}

void Polygons::addPolygon(std::vector<float2> newPoints) {
	bool old;
	cl_int oldNumPts = points.size();
	cl_int oldNumLines = lines.size();
	cl_int lastColor = (lines.size() == 0) ? 0 : lines[lines.size() - 1].color;
	cl_int error = 0;
  
	/* Update point data */
	points.insert(std::end(points), std::begin(newPoints), std::end(newPoints));
	std::vector<cl_int> newColors(newPoints.size(), lastColor + 1);
	pointColors.insert(std::end(pointColors), std::begin(newColors), std::end(newColors));
	numPointsInPolygon.push_back(newPoints.size());
	lasts.push_back(oldNumPts);

	/* Update point buffer size if required, then buffer only the new points. */
	error |= CLFW::getBuffer(pointsBuffer, std::to_string(uid) + "PlygnPts", 
		CLFW::NextPow2(sizeof(float2) * points.size()), old, true);
	error |= CLFW::getBuffer(pointColorsBuffer, std::to_string(uid) + "PlygnPtClrs",
		CLFW::NextPow2(sizeof(cl_int) * points.size()), old, true);

	error |= CLFW::Upload<float2>(newPoints, oldNumPts, pointsBuffer);
	error |= CLFW::Upload<cl_int>(newColors, oldNumPts, pointColorsBuffer);
	assert_cl_error(error);

	/* Update line data */ 
	std::vector<Line> newLines(newPoints.size());
	for (int i = 0; i < newPoints.size(); ++i) {
		Line l;
		l.first = i + oldNumPts;
		l.second = (i == newPoints.size() - 1) ? oldNumPts : i + oldNumPts + 1;
		l.color = lastColor + 1;
		newLines[i] = l;
	}
	lines.insert(std::end(lines), std::begin(newLines), std::end(newLines));

	/* Update line buffer size if required, then buffer only the new lines. */
	error |= CLFW::getBuffer(linesBuffer, std::to_string(uid) + "PlygnLn", 
		CLFW::NextPow2(sizeof(Line) * lines.size()), old, true);
	error |= CLFW::Upload<Line>(newLines, oldNumLines, linesBuffer);

	assert_cl_error(error);
}

void Polygons::removePolygon(int index) {
	/* Requires float2 compaction kernel. */
}

void Polygons::changePolygonColor(int index, float4 color) {

}

void Polygons::movePolygon(int index, glm::mat4 matrix) {
	cl_int error = 0;
	int offset = lasts[index];
	int numPtsToMove = numPointsInPolygon[index];

	error |= Kernels::multiplyM4V2_p(pointsBuffer, numPtsToMove, offset, matrix);
	error |= CLFW::Download<float2>(pointsBuffer, points.size(), points);
	assert_cl_error(error);
}

void Polygons::readMesh(const std::string& filename) {
	std::ifstream in(filename.c_str());
	if (!in) {
		std::cerr << "Failed to read " << filename << std::endl;
		return;
	}

	std::vector<float2> polygonPts;

	float2 pnt;

	while (in >> pnt.x && in >> pnt.y) {
		polygonPts.push_back(pnt);
	}
	in.close();

	addPolygon(polygonPts);
}