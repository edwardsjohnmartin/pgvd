#include "Polylines.h"

int PolyLines::count = 0;

PolyLines::PolyLines(std::vector<std::string> filenames) {
	uid = count;
	count++;

	/* Read files */
	for (std::string s : filenames) readMesh(s);
}

PolyLines::~PolyLines() {
	count--;
}

void PolyLines::clear() {
	points.clear();
	lasts.clear();
	pointColors.clear();
	lines.clear();
	currentColor = 0;
}

void PolyLines::newLine(const float2& p) {
	bool old;
	cl_int error = 0;
	cl_int oldNumPts = points.size();

	/* Add new "lasts" entry for the new polyline. */
	lasts.push_back(1);
	currentColor++;

	/* Update point data. */
	points.push_back(p);
	pointColors.push_back(currentColor);

	//Update point buffer size if required, then buffer up only the new points.
	error |= CLFW::getBuffer(pointsBuffer, std::to_string(uid) + "PlylnPts", 
		sizeof(float2) * CLFW::NextPow2(points.size()), old, true);
	error |= CLFW::getBuffer(pointColorsBuffer, std::to_string(uid) + "PlylnPtClrs",
		sizeof(cl_int) * CLFW::NextPow2(points.size()), old, true);

	error |= CLFW::Upload<float2>((float2)p, oldNumPts, pointsBuffer);
	error |= CLFW::Upload<cl_int>(currentColor, oldNumPts, pointColorsBuffer);
	assert_cl_error(error);
}

void PolyLines::addPoint(const float2& p) {
	bool old;
	cl_int error = 0;
	cl_int oldNumPts = points.size();
	cl_int oldNumLines = lines.size();

	/* Update point/line data */
	points.push_back(p);
	pointColors.push_back(currentColor);
	Line l;
	l.color = currentColor;
	l.first = points.size() - 2;
	l.second = points.size() - 1;
	lines.push_back(l);

	lasts[lasts.size() - 1]++;

	//Update point/line buffer size if required, then buffer up only the new points.
	error |= CLFW::getBuffer(pointsBuffer, std::to_string(uid) + "PlylnPts",
		sizeof(float2) * CLFW::NextPow2(points.size()), old, true);
	error |= CLFW::getBuffer(pointColorsBuffer, std::to_string(uid) + "PlylnPtClrs",
		sizeof(cl_int) * CLFW::NextPow2(points.size()), old, true);
	error |= CLFW::getBuffer(linesBuffer, std::to_string(uid) + "PlylnLn",
		sizeof(Line) * CLFW::NextPow2(lines.size()), old, true);

	error |= CLFW::Upload<float2>((float2)p, oldNumPts, pointsBuffer);
	error |= CLFW::Upload<cl_int>(currentColor, oldNumPts, pointColorsBuffer);
	error |= CLFW::Upload<Line>(l, oldNumLines, linesBuffer);

	assert_cl_error(error);
}

void PolyLines::addLine(const std::vector<float2> newPoints) {
	bool old;
	cl_int oldNumPts = points.size();
	cl_int oldNumLines = lines.size();
	cl_int error = 0;

	/* Add new "lasts" entry for the new polyline. */
	lasts.push_back(newPoints.size());
	currentColor++;

	/* Update point data */
	points.insert(std::end(points), std::begin(newPoints), std::end(newPoints));
	std::vector<cl_int> newColors(newPoints.size(), currentColor);
	pointColors.insert(std::end(pointColors), std::begin(newColors), std::end(newColors));

	/* Update point buffer size if required, then buffer only the new points. */
	error |= CLFW::getBuffer(pointsBuffer, std::to_string(uid) + "PlylnPts",
		sizeof(float2) * CLFW::NextPow2(points.size()), old, true);
	error |= CLFW::getBuffer(pointColorsBuffer, std::to_string(uid) + "PlylnPtClrs",
		sizeof(cl_int) * CLFW::NextPow2(points.size()), old, true);

	error |= CLFW::Upload<float2>(newPoints, oldNumPts, pointsBuffer);
	error |= CLFW::Upload<cl_int>(newColors, oldNumPts, pointColorsBuffer);
	assert_cl_error(error);

	/* Update line data */
	std::vector<Line> newLines(newPoints.size() - 1);
	for (int i = 0; i < newPoints.size() - 1; ++i) {
		Line l;
		l.first = i + oldNumPts;
		l.second = i + oldNumPts + 1;
		l.color = currentColor;
		newLines[i] = l;
	}
	lines.insert(std::end(lines), std::begin(newLines), std::end(newLines));

	/* Update line buffer size if required, then buffer only the new lines. */
	error |= CLFW::getBuffer(linesBuffer, std::to_string(uid) + "PlylnLn",
		CLFW::NextPow2(sizeof(Line) * lines.size()), old, true);
	error |= CLFW::Upload<Line>(newLines, oldNumLines, linesBuffer);

	assert_cl_error(error);
}

void PolyLines::undoLine() {
	if (lasts.size() == 0) return;
	int numToRemove = lasts[lasts.size() - 1];
	points.resize(points.size() - numToRemove);
	pointColors.resize(pointColors.size() - numToRemove);
	lines.resize(lines.size() - (numToRemove - 1));
	lasts.pop_back();
}

void PolyLines::readMesh(const string& filename) {
	ifstream in(filename.c_str());
	if (!in) {
		cerr << "Failed to read " << filename << endl;
		return;
	}

	std::vector<float2> linePts;

	float2 pnt;

	while (in >> pnt.x && in >> pnt.y) {
		linePts.push_back(pnt);
	}
	in.close();

	addLine(linePts);
}

void PolyLines::writeToFile(const string &foldername) {
	ofstream out;
	ofstream subout;
	out.open(foldername + "/" + "files" + ".txt");
	out.clear();
	int first = 0;
	for (int i = 0; i < lasts.size(); ++i) {
		cout << "writing to " << foldername << "/" << i << ".bin" << endl;
		out << foldername << "/" << i << ".bin" << endl;
		subout.open(foldername + "/" + std::to_string(i) + ".bin");
		subout.clear();
		int last = lasts[i];
		for (int j = first; j < first + lasts[i]; j++) {
			subout << points[j].x << " " << points[j].y << endl;
		}
		subout.close();
		first += last;
	}
	out.close();
}