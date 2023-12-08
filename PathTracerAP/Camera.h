#pragma once
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Primitive.h"

using namespace std;

#define RESOLUTION_X 500
#define RESOLUTION_Y 400

struct Ray
{
	glm::vec3 orig;
	glm::vec3 dir;
	glm::vec3 color;
	glm::vec3 hitNormal;
	float t;
	Material::MaterialType mat;
};

class Camera
{
public:

	vector<Ray> rays;
	vector<glm::vec3> image;
	
	void generateRays();
	void generateImage();
};
