#pragma once
#include <vector>
#include "Shaders/Shaders.hpp"
#include "Vector/vec.h"
#include "GLUtilities/gl_utils.h"
#include <glm/glm.hpp>

/* Drawable Objects */
#include "Octree/Octree2.h"
#include "Octree/OctNode.h"
#include "Polylines/Polylines.h"

using namespace std;
extern class Quadtree;
namespace GLUtilities {
	typedef struct Point {
		float4 p;
		float4 color;
	} Point;

	typedef struct SketcherLine {
		Point p1;
		Point p2;
	} SketcherLine;

	typedef struct Box {
		float4 center;
		float scale;
		float4 color;
	} Box;

	class Sketcher
	{
		static Sketcher *s_instance;

		std::vector<Point> points;
		std::vector<SketcherLine> lines;
		std::vector<Box> boxes;

		GLuint pointsVBO;
		GLuint linesVBO;
		GLuint boxesVBO;
		GLuint boxPointsVBO;
		GLuint boxPointIndxVBO;

		GLuint pointsVAO;
		GLuint linesVAO;
		GLuint boxesVAO;
				
	private: 
		void add_internal(vector<OctNode> &o, int i, floatn offset, float scale, float3 color);
		void setupPoints() {
			/* Instances of the "point" struct are added to the pointsVBO. 
				 Having the points data interlaced in memory improves cache coherency. */
			glBindVertexArray(pointsVAO);
			glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
			/* Each point has a position (float4) and color (float4)*/
			glEnableVertexAttribArray(Shaders::sketchProgram->position_id);
			glEnableVertexAttribArray(Shaders::sketchProgram->color_id);
			glVertexAttribPointer(Shaders::sketchProgram->position_id, 4, GL_FLOAT, GL_FALSE, sizeof(Point), 0);
			glVertexAttribPointer(Shaders::sketchProgram->color_id, 4, GL_FLOAT, GL_FALSE, sizeof(Point), (void*)(sizeof(float4)));
			glVertexAttribDivisor(Shaders::sketchProgram->position_id, 1);
			glVertexAttribDivisor(Shaders::sketchProgram->color_id, 1);
			print_gl_error();
		}
		void setupLines() {
			glBindVertexArray(linesVAO);
			glBindBuffer(GL_ARRAY_BUFFER, linesVBO);
			/* Each line contains two points, which each contain a position and color. 
				 Drawing a line is exactly like drawing a point */
			glEnableVertexAttribArray(Shaders::sketchLineProgram->position_id);
			glEnableVertexAttribArray(Shaders::sketchLineProgram->color_id);
			glVertexAttribPointer(Shaders::sketchLineProgram->position_id, 4, 
				GL_FLOAT, GL_FALSE, sizeof(Point), 0);
			glVertexAttribPointer(Shaders::sketchLineProgram->color_id, 4, 
				GL_FLOAT, GL_FALSE, sizeof(Point), (void*)(sizeof(float4)));
			print_gl_error();
		}
		void setupBoxes() {
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
				GL_FALSE, sizeof(Box), (void*)(sizeof(float4)+sizeof(float)));
			print_gl_error();

			/* Setup Instanced Divisors */
			glVertexAttribDivisor(Shaders::boxProgram->offset_id, 1);
			glVertexAttribDivisor(Shaders::boxProgram->scale_id, 1);
			glVertexAttribDivisor(Shaders::boxProgram->color_id, 1);
			print_gl_error();
		}
	public:
		void drawPoints(const glm::mat4& mvMatrix);
		void drawLines(const glm::mat4& mvMatrix);
		void drawBoxes(const glm::mat4& mvMatrix);
		void draw();
		void draw(const glm::mat4& mvMatrix);
		void add(Point p);
		void add(SketcherLine l);
		void add(Box b);
		void add(Quadtree &q); 
		void add(PolyLines &p);
		void clearPoints();
		void clearLines();
		void clearBoxes();
		void clear();
		static Sketcher *instance()
		{
			if (!s_instance)
				s_instance = new Sketcher;
			return s_instance;
		}

		Sketcher() {
			/* Generate buffers */
			glGenBuffers(1, &pointsVBO);
			glGenBuffers(1, &linesVBO);
			glGenBuffers(1, &boxesVBO);
			glGenBuffers(1, &boxPointsVBO);
			glGenBuffers(1, &boxPointIndxVBO);
			print_gl_error();
			assert(glGetError() == GL_NO_ERROR);

			/* Generate attributes */
			glGenVertexArrays(1, &pointsVAO);
			glGenVertexArrays(1, &linesVAO);
			glGenVertexArrays(1, &boxesVAO);
			assert(glGetError() == GL_NO_ERROR);

			/* Associate buffers with attributes */
			setupPoints();
			setupLines();
			setupBoxes();
		}
	};
}
