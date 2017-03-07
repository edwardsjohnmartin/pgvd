#include "OrthoCamera.h"

namespace GLUtilities {
	glm::mat4 OrthoCamera::MV = glm::mat4(1.0);
	glm::mat4 OrthoCamera::IMV = glm::mat4(1.0);

	glm::mat4 OrthoCamera::zoom(glm::vec2 origin, float percent) {
		using namespace glm;
		
		/* Center at origin. Then scale. Then undo translation. */
		MV = glm::translate(MV, vec3(origin, 0.0));
		MV = glm::scale(MV, vec3(percent, percent, 1.0));
		MV = glm::translate(MV, vec3(-origin, 0.0));
		IMV = glm::inverse(MV);
		return MV;
	}
	glm::mat4 OrthoCamera::pan(glm::vec2 displacement) {
		using namespace glm;
		MV = translate(MV, vec3(displacement, 0.0));
		IMV = glm::inverse(MV);
		return MV;
	}
	glm::mat4 OrthoCamera::reset() {
		MV = glm::mat4(1.0);
		IMV = glm::mat4(1.0);
		return MV;
	}
}