#pragma once
#define GLM_FORCE_CUDA

#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Config.h"
#include "Primitive.h"

using namespace std;

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
		Material mat;
		glm::vec3 color;
	} hit_info;

#ifdef ENABLE_VISUALIZER
	struct VisualizerData
	{
		vector<int> hit_voxels;
	} visualizer_data;
#endif
};

class Camera
{
public:

	vector<Ray> rays;
	vector<glm::vec3> image;
	
	void generateRays();
};
