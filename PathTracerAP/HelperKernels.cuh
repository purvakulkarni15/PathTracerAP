#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Primitive.h"
#include "GPUMemoryPool.h"
#include "utility.h"

using namespace Common;
using namespace Camera;
using namespace SceneElements;
using namespace SpatialAcceleration;


void transformModelLauncher(Model* model, GPUMemoryPool<Mesh>* dev_meshes, GPUMemoryPool<Vertex>* dev_vertices)
{
	Mesh* mesh = &dev_meshes->pool[model->mesh_index];
	int size = mesh->vertex_indices.end_index - mesh->vertex_indices.start_index;

	dim3 threads(1024);
	dim3 blocks = (ceil(size / 1024));

	transformModel << <blocks, threads>> > (model, dev_meshes, dev_vertices);
	cudaError_t err = cudaDeviceSynchronize();
}

__global__ void transformModel(Model* model, GPUMemoryPool<Mesh>* dev_meshes, GPUMemoryPool<Vertex>* dev_vertices)
{
	int vert_ind = threadIdx.x + blockDim.x * blockIdx.x;
	Mesh* mesh = &dev_meshes->pool[model->mesh_index];
	int size = mesh->vertex_indices.end_index - mesh->vertex_indices.start_index;
	if (vert_ind > size) return;

	vert_ind = vert_ind + mesh->vertex_indices.start_index;
	dev_vertices->pool[vert_ind].position = transformPosition(dev_vertices->pool[vert_ind].position, model->model_to_world);
}