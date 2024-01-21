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
#include "Experimentation.h"

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

    GPUMemoryPool<Voxel> gpu_memory_pool5;
    render_data.dev_voxel_data = gpu_memory_pool5.getInstance();
    render_data.dev_voxel_data->allocate(scene.voxels);

    GPUMemoryPool<EntityIndex> gpu_memory_pool6;
    render_data.dev_per_voxel_data = gpu_memory_pool6.getInstance();
    render_data.dev_per_voxel_data->allocate(scene.per_voxel_data_pool);

    GPUMemoryPool<Ray> gpu_memory_pool7;
    vector<Ray> rays(RESOLUTION_X * RESOLUTION_Y * SAMPLESX * SAMPLESY);
    render_data.dev_ray_data = gpu_memory_pool7.getInstance();
    render_data.dev_ray_data->allocate(rays);

    GPUMemoryPool<Pixel> gpu_memory_pool8;
    vector<Pixel> image_data(RESOLUTION_X * RESOLUTION_Y);
    render_data.dev_image_data = gpu_memory_pool8.getInstance();
    render_data.dev_image_data->allocate(image_data);

    GPUMemoryPool<int> gpu_memory_pool9;
    vector<int> stencil_data(RESOLUTION_X * RESOLUTION_Y * SAMPLESX * SAMPLESY);
    render_data.dev_stencil = gpu_memory_pool9.getInstance();
    render_data.dev_stencil->allocate(stencil_data);

    GPUMemoryPool<IntersectionData> gpu_memory_pool10;
    vector<IntersectionData> intersection_data(RESOLUTION_X * RESOLUTION_Y*SAMPLESX*SAMPLESY);
    render_data.dev_intersection_data = gpu_memory_pool10.getInstance();
    render_data.dev_intersection_data->allocate(intersection_data);

    GPUMemoryPool<IntersectionData> gpu_memory_pool11;
    render_data.dev_first_intersection_cache = gpu_memory_pool11.getInstance();
    render_data.dev_first_intersection_cache->allocate(intersection_data);

    /*GPUMemoryPool<int> gpu_memory_pool12;
    vector<int> is_intersection_bounding_box(RESOLUTION_X * RESOLUTION_Y);
    render_data.dev_stencil_bounding_box_intersection = gpu_memory_pool12.getInstance();
    render_data.dev_stencil_bounding_box_intersection->allocate(is_intersection_bounding_box);

    GPUMemoryPool<float> gpu_memory_pool13;
    vector<float> t_bounding_box(RESOLUTION_X * RESOLUTION_Y);
    render_data.dev_dist_bounding_box = gpu_memory_pool13.getInstance();
    render_data.dev_dist_bounding_box->allocate(t_bounding_box);*/

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
    render_data.dev_intersection_data->free();
    render_data.dev_first_intersection_cache->free();
    render_data.dev_image_data->free();
    render_data.dev_ray_data->free();
    render_data.dev_stencil->free();
    //render_data.dev_stencil_bounding_box_intersection->free();
    //render_data.dev_dist_bounding_box->free();
}

__inline__ __host__ __device__ 
bool computeRayBoundingBoxIntersection(Ray* ray, BoundingBox* bounding_box, float& t)
{
    float t1 = ray->transformed.dir.x == 0.0f ? FLOAT_MIN : (bounding_box->min.x - ray->transformed.orig.x) * ray->cache.inv_dir.x;
    float t2 = ray->transformed.dir.x == 0.0f ? FLOAT_MAX : (bounding_box->max.x - ray->transformed.orig.x) * ray->cache.inv_dir.x;
    float t3 = ray->transformed.dir.y == 0.0f ? FLOAT_MIN : (bounding_box->min.y - ray->transformed.orig.y) * ray->cache.inv_dir.y;
    float t4 = ray->transformed.dir.y == 0.0f ? FLOAT_MAX : (bounding_box->max.y - ray->transformed.orig.y) * ray->cache.inv_dir.y;
    float t5 = ray->transformed.dir.z == 0.0f ? FLOAT_MIN : (bounding_box->min.z - ray->transformed.orig.z) * ray->cache.inv_dir.z;
    float t6 = ray->transformed.dir.z == 0.0f ? FLOAT_MAX : (bounding_box->max.z - ray->transformed.orig.z) * ray->cache.inv_dir.z;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    if (tmax < 0 || tmin > tmax)
    {
        return false;
    }

    t = tmin;
    return true;
}



__host__ __device__ 
bool computeRayTriangleIntersection(RenderData &render_data, Ray* ray, IntersectionData* hit_info, int itriangle)
{
    Triangle triangle = render_data.dev_triangle_data->pool[itriangle];

    Vertex v0 = render_data.dev_per_vertex_data->pool[triangle.vertex_indices[0]];
    Vertex v1 = render_data.dev_per_vertex_data->pool[triangle.vertex_indices[1]];
    Vertex v2 = render_data.dev_per_vertex_data->pool[triangle.vertex_indices[2]];

    glm::vec3 v0v1 = v1.position - v0.position;
    glm::vec3 v0v2 = v2.position - v0.position;
    glm::vec3 pvec = glm::cross(ray->transformed.dir, v0v2);
    float det = glm::dot(v0v1, pvec);

    if (IS_EQUAL(det, 0.0f)) return false;
    float invDet = 1 / det;

    glm::vec3 tvec = ray->transformed.orig - v0.position;
    float u = glm::dot(tvec, pvec) * invDet;
    if (IS_LESS_THAN(u, 0.0f)|| IS_MORE_THAN(u, 1.0f)) return false;

    glm::vec3 qvec = glm::cross(tvec, v0v1);
    float v = glm::dot(ray->transformed.dir, qvec) * invDet;
    if (IS_LESS_THAN(v, 0.0f) || IS_MORE_THAN(u + v, 1.0f)) return false;

    float t = glm::dot(v0v2, qvec) * invDet;

    if (IS_LESS_THAN(t, 0.0f)) return false;

    glm::vec3 normal = glm::normalize((v0.normal + v1.normal + v2.normal)*(1/3.0f));//glm::cross(v0v1, v0v2);

    //if (det < 0) normal = glm::cross(v0v2, v0v1);
    //else normal = glm::cross(v0v1, v0v2);

    if (hit_info->impact_distance > t)
    {
        hit_info->impact_distance = t;
        hit_info->impact_normal = normal;
    }

    return true;
}

__inline__ __host__ __device__ 
bool computeRayVoxelIntersection(RenderData &render_data, int iray, int ivoxel)
{
    Voxel* voxel= &render_data.dev_voxel_data->pool[ivoxel];
    Ray* ray = &render_data.dev_ray_data->pool[iray];
    IntersectionData* hit_info = &render_data.dev_intersection_data->pool[iray];

    bool isIntersect = false;

    if (voxel->entity_type == EntityType::TRIANGLE)
    {
        for (int i = voxel->entity_index_range.start_index; i < voxel->entity_index_range.end_index; i++)
        {
            int itriangle = render_data.dev_per_voxel_data->pool[i];

            if (computeRayTriangleIntersection(render_data, ray, hit_info, itriangle)) isIntersect = true;
        }
    }
    return isIntersect;
}

__host__ __device__ 
bool computeRayGridIntersection(RenderData &render_data, int iray, int igrid)
{
    Ray* ray = &render_data.dev_ray_data->pool[iray];
    Grid* grid = &render_data.dev_grid_data->pool[igrid];
    BoundingBox* bounding_box;

    if (grid->entity_type == EntityType::MODEL)
    {
        Model* model = &render_data.dev_model_data->pool[grid->entity_index];
        bounding_box = &render_data.dev_mesh_data->pool[model->mesh_index].bounding_box;
    }

    float t_box;
    if (computeRayBoundingBoxIntersection(ray, bounding_box, t_box))
    {
        glm::vec3 grid_intersection_pt = ray->transformed.orig + ray->transformed.dir * t_box;

        if ((grid_intersection_pt.x - bounding_box->min.x) < -EPSILON ||
            (grid_intersection_pt.y - bounding_box->min.y) < -EPSILON ||
            (grid_intersection_pt.z - bounding_box->min.z) < -EPSILON)
        {
            return false;
        }

        Voxel3DIndex ivoxel_3d;
        ivoxel_3d.x = ABS(grid_intersection_pt.x - bounding_box->min.x + EPSILON) / grid->voxel_width.x;
        ivoxel_3d.y = ABS(grid_intersection_pt.y - bounding_box->min.y + EPSILON) / grid->voxel_width.y;
        ivoxel_3d.z = ABS(grid_intersection_pt.z - bounding_box->min.z + EPSILON) / grid->voxel_width.z;

        ivoxel_3d.x = CLAMP(ivoxel_3d.x, 0, GRID_X - 1);
        ivoxel_3d.y = CLAMP(ivoxel_3d.y, 0, GRID_Y - 1);
        ivoxel_3d.z = CLAMP(ivoxel_3d.z, 0, GRID_Z - 1);

        glm::vec3 tMax = glm::vec3(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX);
        glm::vec3 delta = glm::vec3(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX);

        int step_x = ray->transformed.dir.x > 0.0f ? 1 : -1;
        int step_y = ray->transformed.dir.y > 0.0f ? 1 : -1;
        int step_z = ray->transformed.dir.z > 0.0f ? 1 : -1;

        int out_x = ray->transformed.dir.x > 0.0f ? GRID_X : -1;
        int out_y = ray->transformed.dir.y > 0.0f ? GRID_Y : -1;
        int out_z = ray->transformed.dir.z > 0.0f ? GRID_Z : -1;

        int i_next_x = ray->transformed.dir.x > 0.0f ? ivoxel_3d.x + 1 : ivoxel_3d.x;
        float pos_next_x = bounding_box->min.x + i_next_x * grid->voxel_width.x;

        int i_next_y = ray->transformed.dir.y > 0.0f ? ivoxel_3d.y + 1 : ivoxel_3d.y;
        float pos_next_y = bounding_box->min.y + i_next_y * grid->voxel_width.y;

        int i_next_z = ray->transformed.dir.z > 0.0f ? ivoxel_3d.z + 1 : ivoxel_3d.z;
        float pos_next_z = bounding_box->min.z + i_next_z * grid->voxel_width.z;

        if (ray->transformed.dir.x != 0)
        {
            delta.x = ABS(grid->voxel_width.x * ray->cache.inv_dir.x);
            tMax.x = (pos_next_x - grid_intersection_pt.x) * ray->cache.inv_dir.x;
        }

        if (ray->transformed.dir.y != 0)
        {
            delta.y = ABS(grid->voxel_width.y * ray->cache.inv_dir.y);
            tMax.y = (pos_next_y - grid_intersection_pt.y) * ray->cache.inv_dir.y;
        }

        if (ray->transformed.dir.z != 0)
        {
            delta.z = ABS(grid->voxel_width.z * ray->cache.inv_dir.z);
            tMax.z = (pos_next_z - grid_intersection_pt.z) * ray->cache.inv_dir.z;
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

            if (is_intersect && (ABS(ivoxel_cache.x - ivoxel_3d.x) > 2 || ABS(ivoxel_cache.y - ivoxel_3d.y) > 2 || ABS(ivoxel_cache.z - ivoxel_3d.z) > 2))
            {
                return true;
            }

            if (tMax.x < tMax.y && tMax.x < tMax.z)
            {
                ivoxel_3d.x += step_x;
                if (ivoxel_3d.x == out_x || tMax.x >= FLOAT_MAX)
                {
                    return is_intersect;
                }
                tMax.x += delta.x;
            }
            else if (tMax.y < tMax.z)
            {
                ivoxel_3d.y += step_y;
                if (ivoxel_3d.y == out_y || tMax.y >= FLOAT_MAX)
                {
                    return is_intersect;
                }
                tMax.y += delta.y;
            }
            else
            {
                ivoxel_3d.z += step_z;
                if (ivoxel_3d.z == out_z || tMax.z >= FLOAT_MAX)
                {
                    return is_intersect;
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
    IntersectionData* hit_info = &render_data.dev_intersection_data->pool[iray];

    float global_impact_dist = hit_info->impact_distance;
    glm::vec3 global_impact_normal = hit_info->impact_normal;
    Material global_impact_mat = hit_info->impact_mat;

    for (int imodel = 0; imodel < render_data.dev_model_data->size; imodel++)
    {
        Model* model = &render_data.dev_model_data->pool[imodel];

        ray->transformed.orig = transformPosition(ray->base.orig, model->world_to_model);
        ray->transformed.dir = glm::normalize(transformDirection(ray->base.dir, model->world_to_model));
        ray->cache.inv_dir = glm::vec3(1 / ray->transformed.dir.x, 1 / ray->transformed.dir.y, 1 / ray->transformed.dir.z);
        hit_info->impact_distance = FLOAT_MAX;

        if(computeRayGridIntersection(render_data, iray, model->grid_index))
        { 
            glm::vec3 normalized_ray_dir = glm::normalize(ray->transformed.dir);
            glm::vec3 model_coords_intersection = ray->transformed.orig + normalized_ray_dir * hit_info->impact_distance;
            glm::vec3 world_coords_intersection = transformPosition(model_coords_intersection, model->model_to_world);      
            hit_info->impact_distance = glm::length(world_coords_intersection - ray->base.orig);

            if (global_impact_dist > hit_info->impact_distance)
            {
                global_impact_dist = hit_info->impact_distance;
                global_impact_mat = model->mat;
                global_impact_normal = glm::normalize(transformNormal(hit_info->impact_normal, model->model_to_world));
            }
        }
    }

    if (global_impact_dist < FLOAT_MAX)
    {
        hit_info->impact_distance = global_impact_dist;
        hit_info->impact_normal = global_impact_normal;
        hit_info->impact_mat = global_impact_mat;
        return;
    }
}

__global__ 
void shadeRayKernel(int nrays, int iter, RenderData render_data)
{
    int iray = threadIdx.x + blockDim.x * blockIdx.x;

    if (iray >= nrays) return;

    Ray* ray = &render_data.dev_ray_data->pool[iray];
    IntersectionData* hit_info = &render_data.dev_intersection_data->pool[iray];

    if (ray->meta_data.remaining_bounces <= 0)
    {
        ray->color *= glm::vec3(0.01f, 0.01f, 0.01f);
    }

    if (hit_info->impact_distance < FLOAT_MAX)
    {
        glm::vec3 dir = glm::normalize(ray->base.dir);
        glm::vec3 intersection_pt = ray->base.orig + dir * hit_info->impact_distance;

        if (ray->meta_data.remaining_bounces > 0)
        {
            if (hit_info->impact_mat.material_type == Material::MaterialType::DIFFUSE)
            {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, iray, ray->meta_data.remaining_bounces);
                ray->base.dir = calculateRandomDirectionInHemisphere(hit_info->impact_normal, rng);
                ray->base.orig = intersection_pt + 0.1f * hit_info->impact_normal;
                ray->color *= hit_info->impact_mat.color;// *glm::dot(glm::normalize(ray->base.dir), hit_info->impact_normal);
            }
            else if (hit_info->impact_mat.material_type == Material::MaterialType::METAL)
            {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, iray, ray->meta_data.remaining_bounces);
                ray->base.dir = calculateMetalScattering(hit_info->impact_normal, dir, rng);
                ray->base.orig = intersection_pt + 0.1f * hit_info->impact_normal;
                ray->color *= hit_info->impact_mat.color;
            }
            else if (hit_info->impact_mat.material_type == Material::MaterialType::COAT)
            {
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, iray, ray->meta_data.remaining_bounces);
                ray->base.dir = calculateCoatScattering(hit_info->impact_normal, dir, rng);
                ray->base.orig = intersection_pt + 0.1f * hit_info->impact_normal;
                ray->color *= hit_info->impact_mat.color;
            }
            else if (hit_info->impact_mat.material_type == Material::MaterialType::EMISSIVE)
            {
                ray->meta_data.remaining_bounces = 0;
                ray->color *= hit_info->impact_mat.color;
                hit_info->impact_distance = FLOAT_MAX;
                return;
            }
            else if (hit_info->impact_mat.material_type == Material::MaterialType::REFLECTIVE)
            {
                ray->color *= hit_info->impact_mat.color;
                glm::vec3 refected_ray = reflectRay(dir, hit_info->impact_normal);
                ray->base.orig = intersection_pt + 0.1f * hit_info->impact_normal;
                ray->base.dir = refected_ray;
            }
        }
        hit_info->impact_distance = FLOAT_MAX;
    }
    else
    {
        ray->meta_data.remaining_bounces = 0;
        ray->color *= glm::vec3(0.01f, 0.01f, 0.01f);
        hit_info->impact_distance = FLOAT_MAX;
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
    ray->color.x = glm::sqrt(ray->color.x);
    ray->color.y = glm::sqrt(ray->color.y);
    ray->color.z = glm::sqrt(ray->color.z);

    float avg = (1 / (SAMPLESX * SAMPLESY));
    //handle race condition
    render_data.dev_image_data->pool[ray->meta_data.ipixel].color += avg*ray->color;
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

    glm::vec3 camera_orig = glm::vec3(0, 0, 920.0);

    int y = iray / (RESOLUTION_X*SAMPLESX);
    int x = iray % (RESOLUTION_X*SAMPLESX);

    //int y_pix = y / SAMPLESY;
    //int x_pix = x / SAMPLESX;
    //int ipixel = y_pix * RESOLUTION_X + x_pix;
    int ipixel = iray;

    float step_x = 20.0 / (RESOLUTION_X*SAMPLESX);
    float step_y = 16.0 / (RESOLUTION_Y*SAMPLESY);

    float world_x = -10.0 + x * step_x;
    float world_y = -4.0 + y * step_y;
    float world_z = 900.0;

    glm::vec3 pix_pos = glm::vec3(world_x, world_y, world_z);

    render_data.dev_ray_data->pool[iray].base.orig = camera_orig;
    render_data.dev_ray_data->pool[iray].base.dir = glm::vec3(pix_pos - camera_orig);
    render_data.dev_ray_data->pool[iray].color = glm::vec3(1.0f, 1.0f, 1.0f);
    render_data.dev_ray_data->pool[iray].meta_data.remaining_bounces = 5;
    render_data.dev_ray_data->pool[iray].meta_data.ipixel = ipixel;

    render_data.dev_intersection_data->pool[iray].impact_distance = FLOAT_MAX;
    render_data.dev_intersection_data->pool[iray].ipixel = ipixel;
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

    bool is_first_intersection_cached = false;

    for (int iter = 0; iter < ITER; iter++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        bool iterationComplete = false;
        int ibounce = 0;

        nrays = render_data.dev_ray_data->size;
        generateRaysKernel << <blocks, threads >> > (nrays, render_data);
        err = cudaDeviceSynchronize();

        while (!iterationComplete)
        {
            if (ibounce == 0)
            {
                if (is_first_intersection_cached)
                {
                    int size = render_data.dev_first_intersection_cache->size;
                    err = cudaMemcpy(render_data.dev_intersection_data->pool, render_data.dev_first_intersection_cache->pool, sizeof(IntersectionData) * size, cudaMemcpyDeviceToDevice);
                }
                else
                {
                    //cache first intersection;
                    computeRaySceneIntersectionKernel << <blocks, threads >> > (nrays, render_data);
                    err = cudaDeviceSynchronize();
                    //Experiment::computeRaySceneIntersection(nrays, render_data);
                    int size = render_data.dev_intersection_data->size;
                    err = cudaMemcpy(render_data.dev_first_intersection_cache->pool, render_data.dev_intersection_data->pool, sizeof(IntersectionData) * size, cudaMemcpyDeviceToDevice);
                    if (err == cudaSuccess)
                    {
                        is_first_intersection_cached = true;
                    }
                }
            }
            else
            {
                computeRaySceneIntersectionKernel << <blocks, threads >> > (nrays, render_data);
                err = cudaDeviceSynchronize();
                //Experiment::computeRaySceneIntersection(nrays, render_data);
            }
            
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
            ibounce++;
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