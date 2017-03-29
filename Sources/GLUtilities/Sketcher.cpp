#define STB_IMAGE_IMPLEMENTATION
#include "Sketcher.h"


//Allocates pointer to Sketcher instance
GLUtilities::Sketcher *GLUtilities::Sketcher::s_instance = 0;

namespace GLUtilities {
	void Sketcher::add_internal(vector<OctNode> &o, int i, floatn offset, float scale, float3 color)
	{
		if (i != -1 && i >= o.size()) {
			cout << i << endl;
			return;
		}
		Box b = { offset.x, offset.y, 0.0, 1.0, scale, color.x, color.y, color.z, 1.0 };
		boxes.push_back(b);
		if (i != -1 && (o.size() != 1)) {
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

	void Sketcher::uploadImage(string imagePath, string textureName) {
		/* Loads and uploads a texture to the GPU, and adds an entry to textures */
		Texture t = {};
		int comp;
		int forceChannels = 4;
		stbi_set_flip_vertically_on_load(1);
		unsigned char* image = stbi_load(imagePath.c_str(), &t.width, &t.height, &comp, forceChannels);
		assert(image != 0);

		glGenTextures(1, &t.textureId);
		print_gl_error();

		glBindTexture(GL_TEXTURE_2D, t.textureId);
		print_gl_error();

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, t.width, t.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
		print_gl_error();
		
		textures[textureName] = t;

		stbi_image_free(image);
	}
	void Sketcher::uploadObj(string objPath, string objName) {
		obj::Model m = obj::loadModelFromFile("./TestData/images/cube.obj");

		glBindVertexArray(boxesVAO);

		glEnableVertexAttribArray(Shaders::boxProgram->position_id);
		glEnableVertexAttribArray(Shaders::boxProgram->offset_id);
		glEnableVertexAttribArray(Shaders::boxProgram->scale_id);
		glEnableVertexAttribArray(Shaders::boxProgram->color_id);
		print_gl_error();

		float points[] = {
			-1., -1., -1.,  -1., -1., +1.,
			+1., -1., +1.,  +1., -1., -1.,
			-1., +1., -1.,  -1., +1., +1.,
			+1., +1., +1.,  +1., +1., -1.,
		};
		unsigned char indices[] = {
			0, 1, 1, 2, 2, 3, 3, 0,
			0, 4, 1, 5, 2, 6, 3, 7,
			4, 5, 5, 6, 6, 7, 7, 4,
		};
		/* Buffer up box points */
		glBindBuffer(GL_ARRAY_BUFFER, boxPointsVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STREAM_DRAW);
		print_gl_error();

		/* Buffer up box point indices */
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boxPointIndxVBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STREAM_DRAW);
		print_gl_error();

		/* Setup attributes */
		glBindBuffer(GL_ARRAY_BUFFER, boxPointsVBO);
		glVertexAttribPointer(Shaders::boxProgram->position_id, 3, GL_FLOAT,
			GL_FALSE, sizeof(glm::vec3), 0);
		print_gl_error();

		glBindBuffer(GL_ARRAY_BUFFER, boxesVBO);
		glVertexAttribPointer(Shaders::boxProgram->offset_id, 4, GL_FLOAT,
			GL_FALSE, sizeof(Box), 0);
		glVertexAttribPointer(Shaders::boxProgram->scale_id, 1, GL_FLOAT,
			GL_FALSE, sizeof(Box), (void*)sizeof(float4));
		glVertexAttribPointer(Shaders::boxProgram->color_id, 4, GL_FLOAT,
			GL_FALSE, sizeof(Box), (void*)(sizeof(float4) + sizeof(float)));
		print_gl_error();

		/* Setup Instanced Divisors */
		glVertexAttribDivisor(Shaders::boxProgram->offset_id, 1);
		glVertexAttribDivisor(Shaders::boxProgram->scale_id, 1);
		glVertexAttribDivisor(Shaders::boxProgram->color_id, 1);
		print_gl_error();
	}
	void Sketcher::add(Quadtree &q) {
		if (q.nodes.size() == 0 && q.conflicts.size() == 0) return;

		/* Add boxes to represent internal quadtree nodes */
		if (Options::showOctree) {
			floatn center = (q.bb.minimum + q.bb.maxwidth*.5);
			float3 color = { 0.75, 0.75, 0.75 };
			add_internal(q.nodes, 0, center, q.bb.maxwidth * .5, color);
		}

		if (Options::showObjectIntersections)
			for (int i = 0; i < q.conflicts.size(); ++i) {
			Box temp = {};
			floatn min = UnquantizePoint(&q.conflicts[i].origin, &q.bb.minimum, q.resln.width, q.bb.maxwidth);
			temp.scale = (q.conflicts[i].width / (q.resln.width * 2.0)) * q.bb.maxwidth;
			temp.center = { min.x + temp.scale, min.y + temp.scale, 0.0, 1.0 };
			temp.color = { 1.0, 0.0, 0.0, 1.0 };
			add(temp);
		}
		if (Options::showResolutionPoints)
			for (int i = 0; i < q.resolutionPoints.size(); ++i) {
				Point p;
				p.color = { 1.0, 0.0, 0.0, 1.0 };
				p.p = { q.resolutionPoints[i].x, q.resolutionPoints[i].y, 0, 1 };
				add(p);
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
	void Sketcher::add(Plane p) {
		planes.push_back(p);
	}
	void Sketcher::add(PolyLines &p) {
		Color::currentColor = -1;
		
		int first = 0;
		/* For each polygon */
		for (int i = 0; i < p.lasts.size(); ++i) {
			int last = p.lasts[i];
			float3 color = Color::randomColor();
			/* For each point in the line */
			if (Options::showObjectVertices)
				for (int j = first; j < first + last; ++j) {
					Point point;
					point.color = { color.x, color.y, color.z, 1.0 };
					point.p = {p.points[j].x, p.points[j].y, 0.0, 1.0};
					add(point);
				}

			if (Options::showObjects)
				for (int j = first; j < first + last - 1; ++j) {
					Point first, second;
					first.color = { color.x, color.y, color.z, 1.0 };
					second.color = { color.x, color.y, color.z, 1.0 };
					first.p = { p.points[j].x, p.points[j].y, 0.0, 1.0 };
					second.p = { p.points[j + 1].x, p.points[j + 1].y, 0.0, 1.0 };
					SketcherLine l;
					l.p1 = { first };
					l.p2 = { second };
					add(l);
				}
			first += last;
		}
	}

	void Sketcher::add(Polygons &polygons) {
		Color::currentColor = -1;

		int first = 0;
		/* For each polygon */
		for (int i = 0; i < polygons.numPointsInPolygon.size(); ++i) {
			int last = polygons.numPointsInPolygon[i];
			float3 color = Color::randomColor();
			/* For each point in the line */
			if (Options::showObjectVertices)
				for (int j = first; j < first + last; ++j) {
					Point point;
					point.color = { color.x, color.y, color.z, 1.0 };
					point.p = { polygons.points[j].x, polygons.points[j].y, 0.0, 1.0 };
					add(point);
				}

			if (Options::showObjects)
				for (int j = first; j < first + last; ++j) {
					Line line = polygons.lines[j];
					Point first, second;
					first.color = { color.x, color.y, color.z, 1.0 };
					second.color = { color.x, color.y, color.z, 1.0 };
					first.p = { polygons.points[line.first].x, polygons.points[line.first].y, 0.0, 1.0 };
					second.p = { polygons.points[line.second].x, polygons.points[line.second].y, 0.0, 1.0 };
					SketcherLine l;
					l.p1 = { first };
					l.p2 = { second };
					add(l);
				}
			first += last;
		}
	}

	void Sketcher::add(vector<vector<floatn>> polygons) {
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
					point.p = { polygon[j].x, polygon[j].y, 0.0, 1.0 };
					add(point);
				}

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

		Shaders::pointProgram->use();
		glBindVertexArray(pointsVAO);
		glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Point) * points.size(), points.data(), GL_STREAM_DRAW);
		glm::mat4 identity(1.0F);
		// glUniformMatrix4fv(
		//     Shaders::sketchProgram->matrix_id, 1, 0, &(identity[0].x));
		glUniformMatrix4fv(
			Shaders::pointProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
		glUniform1f(Shaders::pointProgram->pointSize_id, 5.0);
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

		Shaders::lineProgram->use();
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
			Shaders::lineProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
		print_gl_error();

		glUniform1f(Shaders::lineProgram->pointSize_id, 10.0);
		print_gl_error();
		glLineWidth(2.);
		glDrawArrays(GL_LINES, 0, 2 * lines.size());
		print_gl_error();
		glBindVertexArray(0);
	}
	void Sketcher::drawBoxes(const glm::mat4& mvMatrix) {
		Shaders::boxProgram->use();

		glBindVertexArray(boxesVAO);
		print_gl_error();

		glBindBuffer(GL_ARRAY_BUFFER, boxesVBO);
		print_gl_error();

		glBufferData(GL_ARRAY_BUFFER, sizeof(Box) * boxes.size(), boxes.data(),
			GL_STREAM_DRAW);
		print_gl_error();

		glUniformMatrix4fv(Shaders::boxProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
		print_gl_error();

		glUniform1f(Shaders::boxProgram->pointSize_id, 10.0);
		print_gl_error();

		glLineWidth(2.0);
		print_gl_error();

		glDrawElementsInstanced(GL_LINES, 12 * 2, GL_UNSIGNED_BYTE, 0, boxes.size());

		print_gl_error();
		glBindVertexArray(0);
	}
	void Sketcher::drawPlanes(const glm::mat4& mvMatrix) {
		glUseProgram(Shaders::planeProgram->program);
		glBindVertexArray(backgroundVAO);
		glBindBuffer(GL_ARRAY_BUFFER, backgroundVBO);
		glUniformMatrix4fv(Shaders::planeProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
		glActiveTexture(GL_TEXTURE0);
		print_gl_error();

		for (int i = 0; i < planes.size(); ++i) {
			glBindTexture(GL_TEXTURE_2D, textures[planes[i].texName].textureId);
			glUniform1i(Shaders::planeProgram->texture_id, 0);
			glUniform3fv(Shaders::planeProgram->offset_uniform_id,
				1, &planes[i].offset[0]);
			glUniform1f(Shaders::planeProgram->width_uniform_id, planes[i].width);
			glUniform1f(Shaders::planeProgram->height_uniform_id, planes[i].height);
			glDrawArrays(GL_TRIANGLES, 0, 6);
			print_gl_error();
		}
	}
	void Sketcher::drawPlane(string texkey, int planeIndex, const glm::mat4& mvMatrix) {
		glUseProgram(Shaders::planeProgram->program);
		glBindVertexArray(backgroundVAO);
		glBindBuffer(GL_ARRAY_BUFFER, backgroundVBO);
		glUniformMatrix4fv(Shaders::planeProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
		glActiveTexture(GL_TEXTURE0);
		print_gl_error();

		glBindTexture(GL_TEXTURE_2D, textures[texkey].textureId);
		glUniform1i(Shaders::planeProgram->texture_id, 0);
		glUniform3fv(Shaders::planeProgram->offset_uniform_id,
			1, &planes[planeIndex].offset[0]);
		glUniform1f(Shaders::planeProgram->width_uniform_id, planes[planeIndex].width);
		glUniform1f(Shaders::planeProgram->height_uniform_id, planes[planeIndex].height);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		print_gl_error();
	}
	void Sketcher::draw() {
		glm::mat4 mvMatrix(1.0);
		drawBoxes(mvMatrix);
		drawLines(mvMatrix);
		drawPoints(mvMatrix);
	}
	void Sketcher::draw(const glm::mat4& mvMatrix) {
		drawBoxes(mvMatrix);
		if (Options::showObjects)
			drawLines(mvMatrix);
		drawPoints(mvMatrix);
	}
}