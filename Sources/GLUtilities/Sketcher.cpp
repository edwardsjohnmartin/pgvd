#include "Sketcher.h"
//Allocates pointer to Sketcher instance
GLUtilities::Sketcher *GLUtilities::Sketcher::s_instance = 0;

namespace GLUtilities {
	void Sketcher::add_internal(vector<OctNode> &o, int i, floatn offset, float scale, float3 color)
	{
		Box b = { offset.x, offset.y, 0.0, 1.0, scale, color.x, color.y, color.z, 1.0 };
		boxes.push_back(b);
		if (i != -1) {
			OctNode current = o[i];
			scale /= 2.0;
			float shift = scale;// / 2.0;
			add_internal(o, current.children[0], { offset.x - shift, offset.y - shift },
				scale, color);
			add_internal(o, current.children[1], { offset.x + shift, offset.y - shift },
				scale, color);
			add_internal(o, current.children[2], { offset.x - shift, offset.y + shift },
				scale, color);
			add_internal(o, current.children[3], { offset.x + shift, offset.y + shift },
				scale, color);
		}
	}

	void Sketcher::add(Quadtree &q) {
		if (q.nodes.size() == 0) return;

		/* Add boxes to represent internal quadtree nodes */
		floatn center = (q.bb.minimum + q.bb.maxwidth*.5);
		float3 color = { 0.75, 0.75, 0.75 };
		add_internal(q.nodes, 0, center, q.bb.maxwidth * .5, color);

		for (int i = 0; i < q.conflicts.size(); ++i) {
			Box temp = {};
			floatn min = UnquantizePoint(&q.conflicts[i].origin, &q.bb.minimum, q.resln.width, q.bb.maxwidth);
			temp.scale = (q.conflicts[i].width / (q.resln.width * 2.0)) * q.bb.maxwidth;
			temp.center = { min.x + temp.scale, min.y + temp.scale, 0.0, 1.0 };
			temp.color = { 1.0, 0.0, 0.0, 1.0 };
			add(temp);
		}

	}
	void Sketcher::add(Point p) {
		points.push_back(p);
	}
	void Sketcher::add(SketcherLine l) {
		lines.push_back(l);
	}
	void Sketcher::add(Box b) {
		boxes.push_back(b);
	}
	void Sketcher::add(PolyLines &p) {
		const vector<vector<floatn>> polygons = p.getPolygons();
		Color::currentColor = -1;
		
		/* For each polygon */
		for (int i = 0; i < polygons.size(); ++i) {
			vector<floatn> polygon = polygons[i];

			float3 color = Color::randomColor();
			/* For each point in the polygon */
			if (Options::showObjectVertices)
				for (int j = 0; j < polygon.size(); ++j) {
					Point point;
					point.color = { color.x, color.y, color.z, 1.0 };
					point.p = {polygon[j].x, polygon[j].y, 0.0, 1.0};
					add(point);
				}

			if (Options::showObjects)
				for (int j = 0; j < polygon.size() - 1; ++j) {
					Point first, second;
					first.color = { color.x, color.y, color.z, 1.0 };
					second.color = { color.x, color.y, color.z, 1.0 };
					first.p = { polygon[j].x, polygon[j].y, 0.0, 1.0 };
					second.p = { polygon[j + 1].x, polygon[j + 1].y, 0.0, 1.0 };
					SketcherLine l;
					l.p1 = { first };
					l.p2 = { second };
					add(l);
				}
		}
		
		//vector<Line> lines = p.getLines();

		//// Get all vertices into a 1D array.
		//for (int i = 0; i < polygons.size(); ++i) {
		//	const vector<floatn>& polygon = polygons[i];
		//	for (int j = 0; j < polygon.size(); ++j) {
		//		points.push_back(polygon[j]);
		//		pointColors.push_back(i);
		//	}
		//}
	}

	void Sketcher::clearPoints() {
		points.clear();
	}
	void Sketcher::clearLines() {
		lines.clear();
	}
	void Sketcher::clearBoxes() {
		boxes.clear();
	}
	void Sketcher::clear() {
		clearPoints();
		clearLines();
		clearBoxes();
	}

	void Sketcher::drawPoints(const glm::mat4& mvMatrix) {
		glEnable(GL_POINT_SMOOTH);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		ignore_gl_error();

		Shaders::sketchProgram->use();
		glBindVertexArray(pointsVAO);
		glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Point) * points.size(), points.data(), GL_STREAM_DRAW);
		glm::mat4 identity(1.0F);
		// glUniformMatrix4fv(
		//     Shaders::sketchProgram->matrix_id, 1, 0, &(identity[0].x));
		glUniformMatrix4fv(
			Shaders::sketchProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
		glUniform1f(Shaders::sketchProgram->pointSize_id, 10.0);
		glDrawArraysInstanced(GL_POINTS, 0, 1, points.size());
		print_gl_error();
		glBindVertexArray(0);
	}
	void Sketcher::drawLines(const glm::mat4& mvMatrix) {
		//lines.clear();
		//lines.push_back({
		//	{{0.0, 0.0, 0.0, 1.0}, {1.0,0.0,0.0,1.0}},
		//	{{0.0,1.0,0.0,1.0}, {1.0,0.0,0.0,1.0}}
		//});
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		print_gl_error();

		Shaders::sketchLineProgram->use();
		glBindVertexArray(linesVAO);
		print_gl_error();

		glBindBuffer(GL_ARRAY_BUFFER, linesVBO);
		print_gl_error();

		glBufferData(GL_ARRAY_BUFFER, sizeof(SketcherLine) * lines.size(), lines.data(), GL_STREAM_DRAW);
		print_gl_error();

		glm::mat4 identity(1.0F);
		// glUniformMatrix4fv(
		//     Shaders::sketchProgram->matrix_id, 1, 0, &(identity[0].x));
		glUniformMatrix4fv(
			Shaders::sketchLineProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
		print_gl_error();

		glUniform1f(Shaders::sketchLineProgram->pointSize_id, 10.0);
		print_gl_error();

		glDrawArrays(GL_LINES, 0, 2 * lines.size());
		print_gl_error();
		glBindVertexArray(0);
	}
	void Sketcher::drawBoxes(const glm::mat4& mvMatrix) {
		Shaders::boxProgram->use();

		glBindVertexArray(boxesVAO);
		glBindBuffer(GL_ARRAY_BUFFER, boxesVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Box) * boxes.size(), boxes.data(),
			GL_STREAM_DRAW);
		glUniformMatrix4fv(Shaders::boxProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
		glUniform1f(Shaders::boxProgram->pointSize_id, 10.0);
		glLineWidth(2.0);
		glDrawElementsInstanced(GL_LINES, 12 * 2, GL_UNSIGNED_BYTE, 0, boxes.size());
		print_gl_error();
		glBindVertexArray(0);
	}
	void Sketcher::draw() {
		glm::mat4 mvMatrix(1.0);
		drawBoxes(mvMatrix);
		drawLines(mvMatrix);
		drawPoints(mvMatrix);
	}
	void Sketcher::draw(const glm::mat4& mvMatrix) {
		drawBoxes(mvMatrix);
		drawLines(mvMatrix);
		drawPoints(mvMatrix);
	}
}