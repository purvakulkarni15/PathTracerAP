#include "Renderer.h"

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>

#include "utility.h"

void Renderer::renderImage()
{
    char bmpHeader[] = {
        'B', 'M',                   // Signature
        0, 0, 0, 0,                  // File size (placeholder)
        0, 0, 0, 0,                  // Reserved
        54, 0, 0, 0,                // Data offset
        40, 0, 0, 0,                // Header size
        static_cast<char>(RESOLUTION_X),   // Image width
        static_cast<char>(RESOLUTION_X >> 8),
        static_cast<char>(RESOLUTION_X >> 16),
        static_cast<char>(RESOLUTION_X >> 24),
        static_cast<char>(RESOLUTION_Y),  // Image height
        static_cast<char>(RESOLUTION_Y >> 8),
        static_cast<char>(RESOLUTION_Y >> 16),
        static_cast<char>(RESOLUTION_Y >> 24),
        1, 0,                       // Number of color planes
        24, 0,                      // Bits per pixel (24 for RGB)
        0, 0, 0, 0,                 // Compression method (0 for no compression)
        0, 0, 0, 0,                 // Image size (placeholder)
        0, 0, 0, 0,                 // Horizontal resolution (pixels per meter)
        0, 0, 0, 0,                 // Vertical resolution (pixels per meter)
        0, 0, 0, 0,                 // Number of colors in the palette
        0, 0, 0, 0                  // Number of important colors
    };

    std::ofstream outFile("Render.bmp", std::ios::binary);

    outFile.write(bmpHeader, sizeof(bmpHeader));

    for (int y = 0; y < RESOLUTION_Y; ++y) {
        for (int x = 0; x < RESOLUTION_X; ++x)
        {
            float div = 1/(float)ITER;
            glm::vec3 color = (render_data.dev_image_data->pool[x + y * RESOLUTION_X].color*div)*255.0f;
            char pixel[] = { color.x,  color.y, color.z};
            outFile.write(pixel, sizeof(pixel));
        }
    }

    int fileSize = 54 + 3 * RESOLUTION_X * RESOLUTION_Y;
    int imageSize = 3 * RESOLUTION_X * RESOLUTION_Y;
    outFile.seekp(2);
    outFile.write(reinterpret_cast<const char*>(&fileSize), 4);
    outFile.seekp(34);
    outFile.write(reinterpret_cast<const char*>(&imageSize), 4);

    outFile.close();
}

void Renderer::allocateOnGPU(Scene &scene)
{
    GPUMemoryPool<Model> gpu_memory_pool0;
    render_data.dev_model_data = gpu_memory_pool0.getInstance();
    render_data.dev_model_data->allocate(scene.models);

    GPUMemoryPool<Mesh> gpu_memory_pool1;
    render_data.dev_mesh_data = gpu_memory_pool1.getInstance();
    render_data.dev_mesh_data->allocate(scene.meshes);

    GPUMemoryPool<Vertex> gpu_memory_pool2;
    render_data.dev_per_vertex_data= gpu_memory_pool2.getInstance();
    render_data.dev_per_vertex_data->allocate(scene.vertices);

    GPUMemoryPool<Triangle> gpu_memory_pool3;
    render_data.dev_triangle_data = gpu_memory_pool3.getInstance();
    render_data.dev_triangle_data->allocate(scene.triangles);

    GPUMemoryPool<Grid> gpu_memory_pool4;
    render_data.dev_grid_data = gpu_memory_pool4.getInstance();
    render_data.dev_grid_data->allocate(scene.grids);

    GPUMemoryPool<IndexRange> gpu_memory_pool5;
    render_data.dev_voxel_data = gpu_memory_pool5.getInstance();
    render_data.dev_voxel_data->allocate(scene.voxels);

    GPUMemoryPool<TriangleIndex> gpu_memory_pool6;
    render_data.dev_per_voxel_data = gpu_memory_pool6.getInstance();
    render_data.dev_per_voxel_data->allocate(scene.per_voxel_data_pool);

    GPUMemoryPool<Ray> gpu_memory_pool7;
    vector<Ray> rays(RESOLUTION_X * RESOLUTION_Y);
    render_data.dev_ray_data = gpu_memory_pool7.getInstance();
    render_data.dev_ray_data->allocate(rays);

    GPUMemoryPool<Pixel> gpu_memory_pool8;
    vector<Pixel> image_data(RESOLUTION_X * RESOLUTION_Y);
    render_data.dev_image_data = gpu_memory_pool8.getInstance();
    render_data.dev_image_data->allocate(image_data);

    GPUMemoryPool<int> gpu_memory_pool9;
    vector<int> stencil_data(RESOLUTION_X * RESOLUTION_Y);
    render_data.dev_stencil = gpu_memory_pool9.getInstance();
    render_data.dev_stencil->allocate(stencil_data);

    printCUDAMemoryInfo();
}

void Renderer::free()
{
    render_data.dev_grid_data->free();
    render_data.dev_mesh_data->free();
    render_data.dev_model_data->free();
    render_data.dev_voxel_data->free();
    render_data.dev_per_vertex_data->free();
    render_data.dev_per_voxel_data->free();
    render_data.dev_triangle_data->free();

    render_data.dev_image_data->free();
    render_data.dev_ray_data->free();
    render_data.dev_stencil->free();
}

__inline__ __host__ __device__ 
bool computeRayBoundingBoxIntersection(Ray* ray, BoundingBox* bounding_box, float& t)
{
    glm::vec3 dir = glm::normalize(ray->points_transformed.dir);
    float inv_x_dir = 1 / dir.x;
    float inv_y_dir = 1 / dir.y;
    float inv_z_dir = 1 / dir.z;

    float t1 = dir.x == 0.0f ? FLOAT_MIN : (bounding_box->min.x - ray->points_transformed.orig.x) * inv_x_dir;
    float t2 = dir.x == 0.0f ? FLOAT_MAX : (bounding_box->max.x - ray->points_transformed.orig.x) * inv_x_dir;
    float t3 = dir.y == 0.0f ? FLOAT_MIN : (bounding_box->min.y - ray->points_transformed.orig.y) * inv_y_dir;
    float t4 = dir.y == 0.0f ? FLOAT_MAX : (bounding_box->max.y - ray->points_transformed.orig.y) * inv_y_dir;
    float t5 = dir.z == 0.0f ? FLOAT_MIN : (bounding_box->min.z - ray->points_transformed.orig.z) * inv_z_dir;
    float t6 = dir.z == 0.0f ? FLOAT_MAX : (bounding_box->max.z - ray->points_transformed.orig.z) * inv_z_dir;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    if (tmax < 0)
    {
        return false;
    }

    if (tmin > tmax)
    {
        return false;
    }

    t = tmin;
    return true;
}

__host__ __device__ 
bool computeRayTriangleIntersection(RenderData &render_data, int iray, int itriangle)
{
    Ray* ray = &render_data.dev_ray_data->pool[iray];
    Triangle triangle = render_data.dev_triangle_data->pool[itriangle];

    Vertex v0 = render_data.dev_per_vertex_data->pool[triangle.vertex_indices[0]];
    Vertex v1 = render_data.dev_per_vertex_data->pool[triangle.vertex_indices[1]];
    Vertex v2 = render_data.dev_per_vertex_data->pool[triangle.vertex_indices[2]];

    //glm::vec3 dir = glm::normalize(ray->points_transformed.end - ray->points_transformed.orig);
    glm::vec3 dir = glm::normalize(ray->points_transformed.dir);

    glm::vec3 v0v1 = v1.position - v0.position;
    glm::vec3 v0v2 = v2.position - v0.position;
    glm::vec3 pvec = glm::cross(dir, v0v2);
    float det = glm::dot(v0v1, pvec);

    if (IS_EQUAL(det, 0.0f)) return false;
    float invDet = 1 / det;

    glm::vec3 tvec = ray->points_transformed.orig - v0.position;
    float u = glm::dot(tvec, pvec) * invDet;
    if (IS_LESS_THAN(u, 0.0f)|| IS_MORE_THAN(u, 1.0f)) return false;

    glm::vec3 qvec = glm::cross(tvec, v0v1);
    float v = glm::dot(dir, qvec) * invDet;
    if (IS_LESS_THAN(v, 0.0f) || IS_MORE_THAN(u + v, 1.0f)) return false;

    float t = glm::dot(v0v2, qvec) * invDet;

    if (IS_LESS_THAN(t, 0.0f)) return false;

    glm::vec3 normal = glm::cross(v0v1, v0v2);

    //if (det < 0) normal = glm::cross(v0v2, v0v1);
    //else normal = glm::cross(v0v1, v0v2);

    if (ray->hit_info.impact_distance > t)
    {
        ray->hit_info.impact_distance = t;
        ray->hit_info.impact_normal = normal;
    }

    return true;
}

__inline__ __host__ __device__ 
bool computeRayVoxelIntersection(RenderData &render_data, int iray, int ivoxel)
{
    IndexRange* voxel_index_range = &render_data.dev_voxel_data->pool[ivoxel];

    bool isIntersect = false;

    for (int i = voxel_index_range->start_index; i < voxel_index_range->end_index; i++)
    {
        int itriangle = render_data.dev_per_voxel_data->pool[i];
        
        if (computeRayTriangleIntersection(render_data, iray, itriangle)) isIntersect = true;
    }
    return isIntersect;
}

__host__ __device__ 
bool computeRayGridIntersection(RenderData &render_data, int iray, int imodel)
{
    Ray* ray = &render_data.dev_ray_data->pool[iray];

    glm::vec3 dir = glm::normalize(ray->points_transformed.dir);

    glm::vec3 inv_dir = glm::vec3(1 / dir.x, 1 / dir.y, 1 / dir.z);

    int igrid = render_data.dev_model_data->pool[imodel].grid_index;

    Grid* grid = &render_data.dev_grid_data->pool[igrid];
    
    int nVoxel = GRID_X * GRID_Y * GRID_Z;
    
    BoundingBox* bounding_box = &render_data.dev_mesh_data->pool[grid->mesh_index].bounding_box;

    float t_box;
    if (computeRayBoundingBoxIntersection(ray, bounding_box, t_box))
    {
        glm::vec3 grid_intersection_pt = ray->points_transformed.orig + dir * t_box;

        if ((grid_intersection_pt.x - bounding_box->min.x) < -EPSILON ||
            (grid_intersection_pt.y - bounding_box->min.y) < -EPSILON ||
            (grid_intersection_pt.z - bounding_box->min.z) < -EPSILON)
        {
            return false;
        }

        Voxel3DIndex ivoxel_3d;
        ivoxel_3d.x = glm::abs(grid_intersection_pt.x - bounding_box->min.x + EPSILON) / grid->voxel_width.x;
        ivoxel_3d.y = glm::abs(grid_intersection_pt.y - bounding_box->min.y + EPSILON) / grid->voxel_width.y;
        ivoxel_3d.z = glm::abs(grid_intersection_pt.z - bounding_box->min.z + EPSILON) / grid->voxel_width.z;

        ivoxel_3d.x = CLAMP(ivoxel_3d.x, 0, GRID_X - 1);
        ivoxel_3d.y = CLAMP(ivoxel_3d.y, 0, GRID_Y - 1);
        ivoxel_3d.z = CLAMP(ivoxel_3d.z, 0, GRID_Z - 1);

        glm::vec3 tMax = glm::vec3(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX);
        glm::vec3 delta = glm::vec3(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX);

        int step_x = dir.x > 0.0f ? 1 : -1;
        int step_y = dir.y > 0.0f ? 1 : -1;
        int step_z = dir.z > 0.0f ? 1 : -1;

        int out_x = dir.x > 0.0f ? GRID_X : -1;
        int out_y = dir.y > 0.0f ? GRID_Y : -1;
        int out_z = dir.z > 0.0f ? GRID_Z : -1;

        int i_next_x = dir.x > 0.0f ? ivoxel_3d.x + 1 : ivoxel_3d.x;
        float pos_next_x = bounding_box->min.x + i_next_x * grid->voxel_width.x;

        int i_next_y = dir.y > 0.0f ? ivoxel_3d.y + 1 : ivoxel_3d.y;
        float pos_next_y = bounding_box->min.y + i_next_y * grid->voxel_width.y;

        int i_next_z = dir.z > 0.0f ? ivoxel_3d.z + 1 : ivoxel_3d.z;
        float pos_next_z = bounding_box->min.z + i_next_z * grid->voxel_width.z;

        if (dir.x != 0)
        {
            delta.x = glm::abs(grid->voxel_width.x * inv_dir.x);
            tMax.x = (pos_next_x - grid_intersection_pt.x) * inv_dir.x;
        }

        if (dir.y != 0)
        {
            delta.y = glm::abs(grid->voxel_width.y * inv_dir.y);
            tMax.y = (pos_next_y - grid_intersection_pt.y) * inv_dir.y;
        }

        if (dir.z != 0)
        {
            delta.z = glm::abs(grid->voxel_width.z * inv_dir.z);
            tMax.z = (pos_next_z - grid_intersection_pt.z) * inv_dir.z;
        }

        Voxel3DIndex ivoxel_cache;
        bool is_intersect = false;

        while (1)
        {
            int ivoxel = grid->voxelIndices.start_index + ivoxel_3d.x + ivoxel_3d.y * GRID_X + ivoxel_3d.z * GRID_X * GRID_Y;
#ifdef ENABLE_VISUALIZER
            int ilast = render_data.visualizer_data.hit_voxels_per_ray.size() - 1;
            render_data.visualizer_data.hit_voxels_per_ray[ilast].push_back(ivoxel);
#endif
            if (computeRayVoxelIntersection(render_data, iray, ivoxel))
            {
                ivoxel_cache = ivoxel_3d;
                is_intersect = true;
            }

            if (is_intersect && (ABS(ivoxel_cache.x -ivoxel_3d.x) > 2 || ABS(ivoxel_cache.y - ivoxel_3d.y) > 2 || ABS(ivoxel_cache.z - ivoxel_3d.z) > 2))
            {
                return true;
            }

            if (tMax.x < tMax.y && tMax.x < tMax.z)
            {
                ivoxel_3d.x += step_x;
                if (ivoxel_3d.x == out_x || tMax.x >= FLOAT_MAX)
                {
                    return false;
                }
                tMax.x += delta.x;
            }
            else if (tMax.y < tMax.z)
            {
                ivoxel_3d.y += step_y;
                if (ivoxel_3d.y == out_y || tMax.y >= FLOAT_MAX)
                {
                    return false;
                }
                tMax.y += delta.y;
            }
            else
            {
                ivoxel_3d.z += step_z;
                if (ivoxel_3d.z == out_z || tMax.z >= FLOAT_MAX)
                {
                    return false;
                }
                tMax.z += delta.z;
            }
        }
    }
}

__global__ 
void computeRaySceneIntersectionKernel(int nrays, RenderData render_data)
{
    int iray = threadIdx.x + blockDim.x * blockIdx.x;

    if (iray >= nrays) return;

    Ray* ray = &render_data.dev_ray_data->pool[iray];

    float global_impact_dist = FLOAT_MAX;
    glm::vec3 global_impact_normal(0.0f);
    Model* global_impact_model;

    for (int imodel = 0; imodel < render_data.dev_model_data->size; imodel++)
    {
        Model* model = &render_data.dev_model_data->pool[imodel];

        ray->points_transformed.orig = transformPosition(ray->points_base.orig, model->world_to_model);
        ray->points_transformed.dir = transformDirection(ray->points_base.dir, model->world_to_model);
        ray->hit_info.impact_distance = FLOAT_MAX;

        computeRayGridIntersection(render_data, iray, imodel);

        if (ray->hit_info.impact_distance < FLOAT_MAX)
        {
            glm::vec3 normalized_ray_dir = glm::normalize(ray->points_transformed.dir);

            glm::vec3 model_coords_intersection = ray->points_transformed.orig + normalized_ray_dir * ray->hit_info.impact_distance;
            glm::vec3 world_coords_intersection = transformPosition(model_coords_intersection, model->model_to_world);      
            ray->hit_info.impact_distance = glm::length(world_coords_intersection - ray->points_base.orig);

            if (global_impact_dist > ray->hit_info.impact_distance)
            {
                global_impact_dist = ray->hit_info.impact_distance;
                global_impact_model = model;
                global_impact_normal = ray->hit_info.impact_normal;
            }
        }
    }

    if (global_impact_dist < FLOAT_MAX)
    {
        ray->hit_info.impact_distance = global_impact_dist;
        ray->hit_info.impact_normal = glm::normalize(transformNormal(global_impact_normal, global_impact_model->model_to_world));
        ray->hit_info.impact_mat = global_impact_model->mat;
        return;
    }
}

__global__ 
void shadeRayKernel(int nrays, int iter, RenderData render_data)
{
    int iray = threadIdx.x + blockDim.x * blockIdx.x;

    if (iray >= nrays) return;

    Ray* ray = &render_data.dev_ray_data->pool[iray];

    if (ray->meta_data.remaining_bounces <= 0) return;

    if (ray->hit_info.impact_distance < FLOAT_MAX)
    {
        glm::vec3 dir = glm::normalize(ray->points_base.dir);
        glm::vec3 intersection_pt = ray->points_base.orig + dir * ray->hit_info.impact_distance;

        if (ray->meta_data.remaining_bounces > 0)
        {
            if (ray->hit_info.impact_mat.material_type == Material::MaterialType::DIFFUSE)
            {
                ray->color *= ray->hit_info.impact_mat.color;
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, iray, ray->meta_data.remaining_bounces);
                glm::vec3 scattered_ray_dir = calculateRandomDirectionInHemisphere(ray->hit_info.impact_normal, rng);
                ray->points_base.orig = intersection_pt + 0.1f * ray->hit_info.impact_normal;
                ray->points_base.dir = scattered_ray_dir;
            }
            else if (ray->hit_info.impact_mat.material_type == Material::MaterialType::EMISSIVE)
            {
                ray->meta_data.remaining_bounces = 0;
                ray->color *= ray->hit_info.impact_mat.color;
                ray->hit_info.impact_distance = FLOAT_MAX;
                return;
            }
            else if (ray->hit_info.impact_mat.material_type == Material::MaterialType::REFLECTIVE)
            {
                ray->color *= ray->hit_info.impact_mat.color;
                glm::vec3 refected_ray = reflectRay(dir, ray->hit_info.impact_normal);
                ray->points_base.orig = intersection_pt + 0.1f * ray->hit_info.impact_normal;
                ray->points_base.dir = refected_ray;
            }
        }
        ray->hit_info.impact_distance = FLOAT_MAX;
    }
    else
    {
        ray->meta_data.remaining_bounces = 0;
        ray->color *= glm::vec3(0.01f, 0.01f, 0.01f);
        ray->hit_info.impact_distance = FLOAT_MAX;
        return;
    }
    ray->meta_data.remaining_bounces--;
}

__global__ 
void gatherImageDataKernel(RenderData render_data)
{
    int iray = threadIdx.x + blockDim.x * blockIdx.x;
    int nrays = render_data.dev_ray_data->size;
    if (iray >= nrays) return;

    Ray* ray = &render_data.dev_ray_data->pool[iray];
    render_data.dev_image_data->pool[ray->meta_data.ipixel].color += ray->color;
}
struct hasTerminated
{
    __host__ __device__
        bool operator()(const int& x)
    {
        return x == 1;
    }
};

__global__ void compactStencilKernel(int nrays, Ray* dev_ray_data, int* dev_stencil)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nrays)
    {
        if (dev_ray_data[index].meta_data.remaining_bounces <= 0)
        {
            dev_stencil[index] = 0;
            return;
        }
        dev_stencil[index] = 1;
    }
}

__global__ 
void generateRaysKernel(int nrays, RenderData render_data)
{
    int iray = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iray >= nrays) return;

    glm::vec3 camera_orig = glm::vec3(0, 0, 1020.0);

    int y = iray / RESOLUTION_X;
    int x = iray % RESOLUTION_X;

    float step_x = 20.0 / RESOLUTION_X;
    float step_y = 16.0 / RESOLUTION_Y;

    float world_x = -10.0 + x * step_x;
    float world_y = -4.0 + y * step_y;
    float world_z = 1000.0;

    glm::vec3 pix_pos = glm::vec3(world_x, world_y, world_z);

    render_data.dev_ray_data->pool[iray].points_base.orig = camera_orig;
    render_data.dev_ray_data->pool[iray].points_base.dir = glm::vec3(pix_pos - camera_orig);
    render_data.dev_ray_data->pool[iray].hit_info.impact_distance = FLOAT_MAX;
    render_data.dev_ray_data->pool[iray].color = glm::vec3(1.0f, 1.0f, 1.0f);
    render_data.dev_ray_data->pool[iray].meta_data.remaining_bounces = 5;
    render_data.dev_ray_data->pool[iray].meta_data.ipixel = iray;
}

__global__ 
void initImageKernel(int nrays, RenderData render_data)
{
    int iray = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iray >= nrays) return;

    render_data.dev_image_data->pool[iray].color = glm::vec3(0.0f, 0.0f, 0.0f);
}

void Renderer::renderLoop()
{
    int nrays = render_data.dev_ray_data->size;
    cudaError_t err;

    dim3 threads(32);
    dim3 blocks = (ceil(nrays / 32));

    auto start_time = std::chrono::high_resolution_clock::now();

    initImageKernel << <blocks, threads >> > (nrays, render_data);
    err = cudaDeviceSynchronize();
   
    for (int iter = 0; iter < ITER; iter++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        bool iterationComplete = false;

        nrays = render_data.dev_ray_data->size;

        generateRaysKernel << <blocks, threads >> > (nrays, render_data);
        err = cudaDeviceSynchronize();

        while (!iterationComplete)
        {
            computeRaySceneIntersectionKernel << <blocks, threads >> > (nrays, render_data);
            err = cudaDeviceSynchronize();

            shadeRayKernel << <blocks, threads >> > (nrays, iter, render_data);
            err = cudaDeviceSynchronize();

            compactStencilKernel << <blocks, threads >> > (nrays, render_data.dev_ray_data->pool, render_data.dev_stencil->pool);
            err = cudaDeviceSynchronize();

            Ray* itr = thrust::stable_partition(thrust::device, render_data.dev_ray_data->pool, render_data.dev_ray_data->pool + nrays, render_data.dev_stencil->pool, hasTerminated());
            int n = itr - render_data.dev_ray_data->pool;
            nrays = n;

            if (nrays == 0)
            {
                iterationComplete = true;
            }
        }
        gatherImageDataKernel << <blocks, threads >> > (render_data);
        
        err = cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Iteration "<<iter+1<<": " << duration.count() << " microseconds" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Full run: " << duration.count() << " microseconds" << std::endl;
}