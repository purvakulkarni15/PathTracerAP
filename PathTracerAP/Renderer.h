#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Camera.h"
#include "Scene.h"
#include "Primitive.h"
#include "GPUMemoryPool.h"

struct RenderData
{
	GPUMemoryPool<Model> *dev_model_data;
	GPUMemoryPool<Mesh> *dev_mesh_data;
	GPUMemoryPool<VertexData> *dev_per_vertex_data;
	GPUMemoryPool<Triangle> *dev_triangle_data;
	GPUMemoryPool<Grid> *dev_grid_data;
	GPUMemoryPool<Range> *dev_voxel_data;
	GPUMemoryPool<int> *dev_per_voxel_data;
	GPUMemoryPool <Ray> *dev_ray_data;
};

class Renderer
{
public:
	__host__ void addScene(Scene &scene);
	__host__ void addRays(vector<Ray> rays);
	__host__ void renderLoop();
	__host__ void renderImage();
	
private:
	RenderData render_data;
};