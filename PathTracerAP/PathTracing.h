#pragma once
#include <Windows.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>      
#include <assimp/scene.h> 
#include <assimp/postprocess.h> 

using namespace std;
#include <fstream>
#include <iostream>

#define RESOLUTION_X 500
#define RESOLUTION_Y 400

#define GRID_X 50
#define GRID_Y 50
#define GRID_Z 50
namespace Scene
{
	class Mesh
	{
	public:
		vector<glm::vec3> vertices;
		vector<glm::vec3> normals;
		vector<glm::vec2> uvs;
		vector<int> indices;
		glm::vec3 bounding_box[2];
		int st, ed;
		void loadMesh(string path);
	private:
		void processMesh(aiMesh* mesh, const aiScene* scene);
		void processNode(aiNode* node, const aiScene* scene);
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

	struct SceneContainer
	{
		vector<Mesh> meshes;
		vector<Material> materials;
	};
}

typedef struct Ray
{
	glm::vec3 orig;
	glm::vec3 dir;
	glm::vec3 color;
	glm::vec3 hitNormal;
	float t;
	Scene::Material::MaterialType mat;
}Ray;

namespace Camera
{
	vector<Ray> rays;
	vector<glm::vec3> image;
	void generateRays();
	void generateImage();
};

class Triangle
{
public:
	glm::vec3* pVerts[3];
	bool intersectRay(Ray& ray);
};

enum GeometryType {
	SPHERE,
	MESH
};

typedef struct Voxel_CPU
{
	vector<int> triangle_indices;
}Voxel_CPU;

typedef struct Grid_CPU
{
	glm::vec3 bounding_box[2];
	vector<Voxel_CPU> voxels;
	int nTriangles;
	GeometryType geometry_type;
}Grid_CPU;

typedef struct Voxel_GPU
{
	int start_index, end_index;
}Voxel_GPU;

typedef struct Grid_GPU
{
	int start_index, end_index;
	glm::vec3 bounding_box[2];
	GeometryType geometry_type;
	glm::mat4 model_to_world;
	glm::mat4 world_to_model;
}Grid_GPU;


namespace UniformGrid
{
	static vector<Grid_CPU> grids;

	void addUniformGrid(Triangle* dev_triangles, int st, int ed, glm::vec3 bounding_box[2]);
	bool intersectVoxel(Ray& ray, Voxel_GPU& voxel);
	bool intersectBoundingBox(Ray& ray, glm::vec3 bounding_box[2]);
	bool intersectGrid(Ray& ray, Grid_GPU& grid);
};


namespace GPU
{
	glm::vec3* dev_vertices;
	int nVertices;
	glm::vec3* dev_normals;
	glm::vec2* dev_uvs;
	Triangle* dev_triangles;
	int nTriangles;
	Grid_GPU* dev_grids;
	int nGrid;
	Voxel_GPU* dev_voxels;
	int* dev_triangleIndices;
	Ray* rays;
	int nRay;

	//Memory Allocation
	void allocateMemoryForRays(vector<Ray>& rays_in);
	void allocateMemoryForMeshes(vector<Scene::Mesh>& meshes);
	//void deallocateMemoryForMeshes();
	void allocateMemoryForGrids(vector<Grid_CPU>& grids);
	//void deallocateMemoryForGrids();

	//Kernels
	void computeRayMeshIntersection();
};