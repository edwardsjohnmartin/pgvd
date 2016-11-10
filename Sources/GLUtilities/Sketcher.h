#pragma once
#include <vector>
#include "../Shaders/Shaders.hpp"
#include "../../SharedSources/Vector/vec.h"
#include "../GLUtilities/gl_utils.h"
#include <glm/glm.hpp>
using namespace std;
namespace GLUtilities {
    typedef struct Point {
        float4 p;
        float4 color;
    } Point;

    typedef struct Line {
        Point p1;
        Point p2;
    } Line;

    typedef struct Box {
        float4 min;
        float width;
        float4 color[3];
    } Box;

    class Sketcher
    {
        std::vector<Point> points;
        std::vector<Line> lines;
        std::vector<Box> boxes;

        GLuint VBO;
        GLuint linesVBO;
        GLuint boxesVBO;

        GLuint pointsVAO;
        GLuint linesVAO;
        GLuint boxesVAO;

        static Sketcher *s_instance;

        Sketcher() {
            /* Generate buffers */
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &linesVBO);
            glGenBuffers(1, &boxesVBO);
            assert(glGetError() == GL_NO_ERROR);

            /* Generate attributes */
            glGenVertexArrays(1, &pointsVAO);
            glGenVertexArrays(1, &linesVAO);
            glGenVertexArrays(1, &boxesVAO);
            assert(glGetError() == GL_NO_ERROR);

            /* Associate buffers with attributes */
            glBindVertexArray(pointsVAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glEnableVertexAttribArray(Shaders::sketchProgram->position_id);
            glEnableVertexAttribArray(Shaders::sketchProgram->color_id);
            glVertexAttribPointer(Shaders::sketchProgram->position_id, 4, GL_FLOAT, GL_FALSE, sizeof(Point), 0);
            glVertexAttribPointer(Shaders::sketchProgram->color_id, 4, GL_FLOAT, GL_FALSE, sizeof(Point), (void*)(sizeof(float4)));
            glVertexAttribDivisor(Shaders::sketchProgram->position_id, 1);
            glVertexAttribDivisor(Shaders::sketchProgram->color_id, 1);
            assert(glGetError() == GL_NO_ERROR);

            glBindVertexArray(linesVAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glEnableVertexAttribArray(Shaders::sketchProgram->position_id);
            glEnableVertexAttribArray(Shaders::sketchProgram->color_id);
            glVertexAttribPointer(Shaders::sketchProgram->position_id, 4, GL_FLOAT, GL_FALSE, sizeof(Point), 0);
            glVertexAttribPointer(Shaders::sketchProgram->color_id, 4, GL_FLOAT, GL_FALSE, sizeof(Point), (void*)(sizeof(float4)));
            assert(glGetError() == GL_NO_ERROR);


            //Line l1 = {
            //    {
            //        {-.5, -.5, 0.0, 1.0},
            //        {1.0, 0.0, 0.0, 1.0},
            //    },
            //    {
            //        { .5, .5, 0.0, 1.0 },
            //        { 0.0, 1.0, 0.0, 1.0 },
            //    },
            //};
            //Line l2 = {
            //    {
            //        { -.5, .5, 0.0, 1.0 },
            //        { 1.0, 0.0, 0.0, 1.0 },
            //    },
            //    {
            //        { .5, -.5, 0.0, 1.0 },
            //        { 0.0, 1.0, 0.0, 1.0 },
            //    },
            //};
            //lines.push_back(l2);
            //lines.push_back(l1);
            //points.push_back(l1.p1);
            //points.push_back(l1.p2);
            //TODO: Implement boxes.
            //glBindVertexArray(boxesVAO);
            //glBindBuffer(GL_ARRAY_BUFFER, boxesVBO);
        }
    public:
      void drawPoints(const glm::mat4& mvMatrix) {
        print_gl_error();
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        ignore_gl_error();

        //is_programme_valid(Shaders::sketchProgram->program);

        print_gl_error();
        Shaders::sketchProgram->use();
        print_gl_error();
        glBindVertexArray(pointsVAO);
        print_gl_error();
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        print_gl_error();
        glBufferData(GL_ARRAY_BUFFER, sizeof(Point) * points.size(), points.data(), GL_STREAM_DRAW);
        print_gl_error();
        glm::mat4 identity(1.0F);
        print_gl_error();
        // glUniformMatrix4fv(
        //     Shaders::sketchProgram->matrix_id, 1, 0, &(identity[0].x));
        glUniformMatrix4fv(
            Shaders::sketchProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
        print_gl_error();
        glUniform1f(Shaders::sketchProgram->pointSize_id, 10.0);
        print_gl_error();
        glDrawArraysInstanced(GL_POINTS, 0, 1, points.size());
        print_gl_error();
        glBindVertexArray(0);
      }
      void drawLines(const glm::mat4& mvMatrix) {
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

        Shaders::sketchProgram->use();
        glBindVertexArray(linesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Line) * lines.size(), lines.data(), GL_STREAM_DRAW);
        glm::mat4 identity(1.0F);
        // glUniformMatrix4fv(
        //     Shaders::sketchProgram->matrix_id, 1, 0, &(identity[0].x));
        glUniformMatrix4fv(
            Shaders::sketchProgram->matrix_id, 1, 0, &(mvMatrix[0].x));
        glUniform1f(Shaders::sketchProgram->pointSize_id, 10.0);
        glDrawArrays(GL_LINES, 0, 2 * lines.size());
        print_gl_error();
        glBindVertexArray(0);
      }
        void drawBoxes(const glm::mat4& mvMatrix) {
            //cout << "drawing boxes!" << endl;
        }
        void draw(const glm::mat4& mvMatrix) {
            drawPoints(mvMatrix);
            drawLines(mvMatrix);
            drawBoxes(mvMatrix);
        }
        void add(Point p) {
            points.push_back(p);
        }
        void add(Line l) {
            lines.push_back(l);
        }
        void add(Box b) {
            boxes.push_back(b);
        }
        void clearPoints() {
            points.clear();
        }
        void clearLines() {
            lines.clear();
        }
        void clearBoxes() {
            boxes.clear();
        }
        void clear() {
            clearPoints();
            clearLines();
            clearBoxes();
        }
        static Sketcher *instance()
        {
            if (!s_instance)
                s_instance = new Sketcher;
            return s_instance;
        }
    };
}
