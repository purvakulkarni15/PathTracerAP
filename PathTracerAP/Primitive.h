#pragma once
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
		REFRACTIVE
	} material_type;

	float refractive_index;
	float reflectivity;
	glm::vec3 color;
};

struct Mesh
{
	Range vertexDataIndices;
	Range triangleDataIndices;
	glm::vec3 bounding_box[2];
};
