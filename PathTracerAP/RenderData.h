#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Primitive.h"
#include "GPUMemoryPool.h"

using namespace Common;
using namespace Camera;
using namespace SceneElements;
using namespace SpatialAcceleration;

static class RenderData
{
	public:
		static GPUMemoryPool<Model>* dev_models;
		static GPUMemoryPool<Mesh>* dev_meshes;
		static GPUMemoryPool<Vertex>* dev_vertices;
		static GPUMemoryPool<Triangle>* dev_triangles;
		static Grid dev_grid;
		static GPUMemoryPool<Voxel>* dev_voxels;
		static GPUMemoryPool<EntityIndex>* dev_per_voxel_data;
		static GPUMemoryPool <Ray>* dev_rays;
		static GPUMemoryPool<IntersectionData>* dev_intersections;
		static GPUMemoryPool<IntersectionData>* dev_first_intersections_cache;
		static GPUMemoryPool <Pixel>* dev_image;
		static GPUMemoryPool <int>* dev_stencil;

	static void allocateModels(vector<Model> &host_models)
	{
		GPUMemoryPool<Model> gpu_memory_pool;
		dev_models = gpu_memory_pool.getInstance();
		dev_models->allocate(host_models);
	}

	static void allocateMeshes(vector<Mesh>& host_meshes)
	{
		GPUMemoryPool<Mesh> gpu_memory_pool;
		dev_meshes = gpu_memory_pool.getInstance();
		dev_meshes->allocate(host_meshes);
	}

	static void allocateVertices(vector<Vertex>& host_vertices)
	{
		GPUMemoryPool<Vertex> gpu_memory_pool;
		dev_vertices = gpu_memory_pool.getInstance();
		dev_vertices->allocate(host_vertices);
	}

	static void allocateTriangles(vector<Triangle>& host_triangles)
	{
		GPUMemoryPool<Triangle> gpu_memory_pool;
		dev_triangles = gpu_memory_pool.getInstance();
		dev_triangles->allocate(host_triangles);
	}

	static void allocateVoxels(vector<Voxel>& host_voxels)
	{
		GPUMemoryPool<Voxel> gpu_memory_pool;
		dev_voxels = gpu_memory_pool.getInstance();
		dev_voxels->allocate(host_voxels);
	}

	static void allocatePerVoxelData(vector<EntityIndex>& host_per_voxel_data)
	{
		GPUMemoryPool<EntityIndex> gpu_memory_pool;
		dev_per_voxel_data = gpu_memory_pool.getInstance();
		dev_per_voxel_data->allocate(host_per_voxel_data);
	}

	static void allocateRays(vector<Ray>& host_rays)
	{
		GPUMemoryPool<Ray> gpu_memory_pool;
		dev_rays = gpu_memory_pool.getInstance();
		dev_rays->allocate(host_rays);
	}

	static void allocateIntersectionData(vector<IntersectionData>& host_intersections)
	{
		GPUMemoryPool<IntersectionData> gpu_memory_pool;
		dev_intersections = gpu_memory_pool.getInstance();
		dev_intersections->allocate(host_intersections);
	}

	static void allocateIntersectionDataCache(vector<IntersectionData>& host_intersections_cache)
	{
		GPUMemoryPool<IntersectionData> gpu_memory_pool;
		dev_first_intersections_cache = gpu_memory_pool.getInstance();
		dev_first_intersections_cache->allocate(host_intersections_cache);
	}

	static void allocateStencil(vector<int>& host_stencil)
	{
		GPUMemoryPool<int> gpu_memory_pool;
		dev_stencil = gpu_memory_pool.getInstance();
		dev_stencil->allocate(host_stencil);
	}
};
