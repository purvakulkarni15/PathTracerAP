#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>      
#include <assimp/scene.h> 
#include <assimp/postprocess.h> 

#include "Primitive.h"

using namespace std;

#define GRID_X 50
#define GRID_Y 50
#define GRID_Z 50

struct Grid
{
	Range voxelIndices;
	int mesh_index;
};

struct Model
{
	int grid_index;
	int mesh_index;
	glm::mat4 model_to_world;
	glm::mat4 world_to_model;
	Material mat;
};

class Scene
{
public:
	Scene(string config);

	vector<Model> models;
	vector<Mesh> meshes;
	vector<VertexData> vertex_data_pool;
	vector<Triangle> triangles;
	vector<Grid> grids;
	vector<Range> voxels;
	vector<int> per_voxel_data_pool;

private:
	void addMesh(string path, Mesh& mesh);
	void generateUniformGrids();
	void processMesh(aiMesh* ai_mesh, Mesh& mesh, const aiScene* scene);
	void processNode(aiNode* node, Mesh& mesh, const aiScene* scene);
};

