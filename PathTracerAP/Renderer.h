#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Scene.h"
#include "Primitive.h"
#include "Config.h"
#include "GPUMemoryPool.h"

using namespace Common;
using namespace Camera;
using namespace SceneElements;
using namespace SpatialAcceleration;

class Renderer
{
public:
	__host__ void allocateOnGPU(Scene &scene);
	__host__ void renderLoop();
	__host__ void renderImage();
	__host__ void free();

	RenderData render_data;
};