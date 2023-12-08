#pragma once
#include "Camera.h"
#include "Scene.h"
#include "Primitive.h"
#include "GPUMemoryPool.h"

class Renderer
{
public:
	void addScene(Scene &scene);
	void addRays(vector<Ray> rays);

	void renderImage();
	
private:
	bool computeRayGridIntersection(Ray& ray, Grid& grid);
	bool computeRayTriangleIntersection(Triangle tri, Ray& ray);
	bool computeRayVoxelIntersection(Ray& ray, Range& voxel);
	bool computeRayBoundingBoxIntersection(Ray& ray, glm::vec3 bounding_box[2]);
	void computeRaySceneIntersection();

	GPUMemoryPool<Mesh>* dev_meshes;
	GPUMemoryPool<VertexData>* dev_vertexDataArr;
	GPUMemoryPool<Triangle>* dev_triangles;
	GPUMemoryPool<Grid>* dev_grids;
	GPUMemoryPool<Range>* dev_voxels;
	GPUMemoryPool<int>* dev_perVoxelDataPool;
	GPUMemoryPool <Ray>* dev_rays;
};