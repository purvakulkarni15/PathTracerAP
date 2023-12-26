#pragma once
#include <iostream>
#include <vector>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Config.h"

namespace Camera
{
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
}



struct VertexData
{
	glm::vec3 vertex;
	glm::vec3 normal;
	glm::vec2 uv;
};

struct Triangle
{
	int indices[3];
};

struct TriangleIndex
{
	int index;
};

struct VoxelIndex
{
	int index;
};

struct Range
{
	int start_index;
	int end_index;
};

struct Material
{
	enum MaterialType {
		DIFFUSE,
		SPECULAR,
		REFLECTIVE,
		REFRACTIVE,
		EMISSIVE
	} material_type;

	float refractive_index;
	float reflectivity;
	glm::vec3 color;
};

struct Bounding_Box
{
	glm::vec3 min, max;
	Bounding_Box()
	{
		//Initialize bounding box
		min.x = FLOAT_MAX;
		min.y = FLOAT_MAX;
		min.z = FLOAT_MAX;

		max.x = FLOAT_MIN;
		max.y = FLOAT_MIN;
		max.z = FLOAT_MIN;
	}

	void update(glm::vec3 vertex)
	{
		min.x = min.x > vertex.x ? vertex.x : min.x;
		min.y = min.y > vertex.y ? vertex.y : min.y;
		min.z = min.z > vertex.z ? vertex.z : min.z;

		max.x = max.x < vertex.x ? vertex.x : max.x;
		max.y = max.y < vertex.y ? vertex.y : max.y;
		max.z = max.z < vertex.z ? vertex.z : max.z;
	}
};

struct Mesh
{
	Range vertexDataIndices;
	Range triangleDataIndices;
	Bounding_Box bounding_box;
};
