#pragma once
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace GLUtilities {
	class OrthoCamera {
		private:
			static glm::vec2 position;
		public: 
			static glm::mat4 MV;
			static glm::mat4 IMV;
			static glm::mat4 zoom(glm::vec2 origin, float scale);
			static glm::mat4 pan(glm::vec2 displacement);
			static glm::mat4 reset();
	};
}