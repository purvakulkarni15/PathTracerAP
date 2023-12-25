#pragma once
#define GLM_FORCE_CUDA

#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Config.h"
#include "Primitive.h"

using namespace std;

struct Pixel
{
	glm::vec3 color;
};

struct Ray
{
	struct Points
	{
		glm::vec3 orig;
		glm::vec3 end;
	} points_base, points_transformed;

	struct HitInfo
	{
		float t;
		glm::vec3 impact_normal;
		Material impact_mat;
	} hit_info;

	struct MetaData
	{
		int ipixel;
		int remainingBounces;
	}meta_data;

	glm::vec3 color;
};

class Camera
{
public:

	vector<Ray> rays;
	vector<glm::vec3> image;
	
	void generateRays();
};
