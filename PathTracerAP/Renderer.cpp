#include "Renderer.h"
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define IS_EQUAL(x, y) (ABS((x) - (y)) < EPSILON)
#define IS_LESS_THAN(x, y) ((x) < (y) - EPSILON)
#define IS_MORE_THAN(x, y) ((x) > (y) + EPSILON)
#define CLAMP(value, min, max) ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f

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

void Renderer::addScene(Scene &scene)
{
    render_data.dev_model_data = GPUMemoryPool<Model>::getInstance();
    render_data.dev_model_data->allocate(scene.models);

    render_data.dev_mesh_data = GPUMemoryPool<Mesh>::getInstance();
    render_data.dev_mesh_data->allocate(scene.meshes);

    render_data.dev_per_vertex_data= GPUMemoryPool<VertexData>::getInstance();
    render_data.dev_per_vertex_data->allocate(scene.vertex_data_pool);

    render_data.dev_triangle_data = GPUMemoryPool<Triangle>::getInstance();
    render_data.dev_triangle_data->allocate(scene.triangles);

    render_data.dev_grid_data = GPUMemoryPool<Grid>::getInstance();
    render_data.dev_grid_data->allocate(scene.grids);

    render_data.dev_voxel_data = GPUMemoryPool<Range>::getInstance();
    render_data.dev_voxel_data->allocate(scene.voxels);

    render_data.dev_per_voxel_data = GPUMemoryPool<TriangleIndex>::getInstance();
    render_data.dev_per_voxel_data->allocate(scene.per_voxel_data_pool);
}

void Renderer::addRays(vector<Ray> rays)
{
    render_data.dev_ray_data = GPUMemoryPool<Ray>::getInstance();
    render_data.dev_ray_data->allocate(rays);

    vector<Pixel> image_data(RESOLUTION_X*RESOLUTION_Y);
    for (int i = 0; i < image_data.size(); i++)
    {
        image_data[i].color.x = 1.0f;
        image_data[i].color.y = 1.0f;
        image_data[i].color.z = 1.0f;
    }
    render_data.dev_image_data = GPUMemoryPool<Pixel>::getInstance();
    render_data.dev_image_data->allocate(image_data);

    vector<int> stencil_data(RESOLUTION_X * RESOLUTION_Y);
    render_data.dev_stencil = GPUMemoryPool<int>::getInstance();
    render_data.dev_stencil->allocate(stencil_data);
}

__host__ __device__ bool computeRayBoundingBoxIntersection(Ray* ray, Bounding_Box* bounding_box, float& t)
{
    glm::vec3 dir = glm::normalize(ray->points_transformed.end - ray->points_transformed.orig);

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

__host__ __device__ bool computeRayTriangleIntersection(RenderData &render_data, int iray, int itriangle, int imodel)
{
    Ray* ray = &render_data.dev_ray_data->pool[iray];
    Triangle triangle = render_data.dev_triangle_data->pool[itriangle];

    VertexData v0 = render_data.dev_per_vertex_data->pool[triangle.indices[0]];
    VertexData v1 = render_data.dev_per_vertex_data->pool[triangle.indices[1]];
    VertexData v2 = render_data.dev_per_vertex_data->pool[triangle.indices[2]];

    glm::vec3 dir = glm::normalize(ray->points_transformed.end - ray->points_transformed.orig);

    glm::vec3 v0v1 = v1.vertex - v0.vertex;
    glm::vec3 v0v2 = v2.vertex - v0.vertex;
    glm::vec3 pvec = glm::cross(dir, v0v2);
    float det = glm::dot(v0v1, pvec);

    if (IS_EQUAL(det, 0.0f)) return false;
    float invDet = 1 / det;

    glm::vec3 tvec = ray->points_transformed.orig - v0.vertex;
    float u = glm::dot(tvec, pvec) * invDet;
    if (IS_LESS_THAN(u, 0.0f)|| IS_MORE_THAN(u, 1.0f)) return false;

    glm::vec3 qvec = glm::cross(tvec, v0v1);
    float v = glm::dot(dir, qvec) * invDet;
    if (IS_LESS_THAN(v, 0.0f) || IS_MORE_THAN(u + v, 1.0f)) return false;

    float t = glm::dot(v0v2, qvec) * invDet;

    if (IS_LESS_THAN(t, 0.0f)) return false;

    glm::vec3 model_coords_intersection = ray->points_transformed.orig + dir * t;
    glm::vec3 world_coords_intersection = glm::vec3(render_data.dev_model_data->pool[imodel].model_to_world * glm::vec4(model_coords_intersection, 1.0f));
    glm::vec3 model_coords_normal = glm::normalize(glm::cross(v0v1, v0v2));
    //if (det < 0) normal = glm::cross(v0v2, v0v1);
    //else normal = glm::cross(v0v1, v0v2);

    glm::vec3 end_normal = model_coords_intersection + model_coords_normal * 10.0f;
    glm::vec3 world_end_normal = glm::vec3(render_data.dev_model_data->pool[imodel].model_to_world * glm::vec4(end_normal, 1.0f));

    glm::vec3 world_coords_normal = glm::normalize(glm::vec3(world_end_normal - world_coords_intersection));

    t = glm::length(world_coords_intersection - ray->points_base.orig);

    if (ray->hit_info.t > t)
    {
        ray->hit_info.t = t;
        ray->hit_info.impact_normal = world_coords_normal;
        ray->hit_info.impact_mat = render_data.dev_model_data->pool[imodel].mat;
    }

    return true;
}

__host__ __device__ bool computeRayVoxelIntersection(RenderData &render_data, int iray, int ivoxel, int imodel)
{
    Range* voxel = &render_data.dev_voxel_data->pool[ivoxel];

    bool isIntersect = false;

    for (int i = voxel->start_index; i < voxel->end_index; i++)
    {
        int itriangle = render_data.dev_per_voxel_data->pool[i].index;
        
        if (computeRayTriangleIntersection(render_data, iray, itriangle, imodel)) isIntersect = true;
    }
    return isIntersect;
}

__host__ __device__ bool computeRayGridIntersection(RenderData &render_data, int iray, int imodel)
{
    Ray* ray = &render_data.dev_ray_data->pool[iray];

    glm::vec3 dir = glm::normalize(ray->points_transformed.end - ray->points_transformed.orig);
    
    glm::vec3 inv_dir = glm::vec3(1 / dir.x, 1 / dir.y, 1 / dir.z);

    int igrid = render_data.dev_model_data->pool[imodel].grid_index;

    Grid* grid = &render_data.dev_grid_data->pool[igrid];
    
    int nVoxel = GRID_X * GRID_Y * GRID_Z;
    
    Bounding_Box* bounding_box = &render_data.dev_mesh_data->pool[grid->mesh_index].bounding_box;

    float t_box;
    if (computeRayBoundingBoxIntersection(ray, bounding_box, t_box))
    {
        glm::vec3 grid_intersection_pt = ray->points_transformed.orig + dir * t_box;

        float x_width = bounding_box->max.x - bounding_box->min.x;
        float y_width = bounding_box->max.y - bounding_box->min.y;
        float z_width = bounding_box->max.z - bounding_box->min.z;

        float x_cell_width = x_width / GRID_X;
        float y_cell_width = y_width / GRID_Y;
        float z_cell_width = z_width / GRID_Z;

        if ((grid_intersection_pt.x - bounding_box->min.x) < -EPSILON ||
            (grid_intersection_pt.y - bounding_box->min.y) < -EPSILON ||
            (grid_intersection_pt.z - bounding_box->min.z) < -EPSILON)
        {
            return false;
        }

        Voxel3D ivoxel_3d;
        ivoxel_3d.x = glm::abs(grid_intersection_pt.x - bounding_box->min.x + EPSILON) / x_cell_width;
        ivoxel_3d.y = glm::abs(grid_intersection_pt.y - bounding_box->min.y + EPSILON) / y_cell_width;
        ivoxel_3d.z = glm::abs(grid_intersection_pt.z - bounding_box->min.z + EPSILON) / z_cell_width;

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
        float pos_next_x = bounding_box->min.x + i_next_x * x_cell_width;

        int i_next_y = dir.y > 0.0f ? ivoxel_3d.y + 1 : ivoxel_3d.y;
        float pos_next_y = bounding_box->min.y + i_next_y * y_cell_width;

        int i_next_z = dir.z > 0.0f ? ivoxel_3d.z + 1 : ivoxel_3d.z;
        float pos_next_z = bounding_box->min.z + i_next_z * z_cell_width;

        if (dir.x != 0)
        {
            delta.x = glm::abs(x_cell_width * inv_dir.x);
            tMax.x = (pos_next_x - grid_intersection_pt.x) * inv_dir.x;
        }

        if (dir.y != 0)
        {
            delta.y = glm::abs(y_cell_width * inv_dir.y);
            tMax.y = (pos_next_y - grid_intersection_pt.y) * inv_dir.y;
        }

        if (dir.z != 0)
        {
            delta.z = glm::abs(z_cell_width * inv_dir.z);
            tMax.z = (pos_next_z - grid_intersection_pt.z) * inv_dir.z;
        }
        while (1)
        {
            int ivoxel = grid->voxelIndices.start_index + ivoxel_3d.x + ivoxel_3d.y * GRID_X + ivoxel_3d.z * GRID_X * GRID_Y;
#ifdef ENABLE_VISUALIZER
            int ilast = render_data.visualizer_data.hit_voxels_per_ray.size() - 1;
            render_data.visualizer_data.hit_voxels_per_ray[ilast].push_back(ivoxel);
#endif
            if (computeRayVoxelIntersection(render_data, iray, ivoxel, imodel))
            {
                //return true;
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

__global__ void computeRaySceneIntersection_kernel(int nrays, RenderData render_data)
{
    int iray = threadIdx.x + blockDim.x * blockIdx.x;

    if (iray >= nrays) return;

    Ray* ray = &render_data.dev_ray_data->pool[iray];

    for (int imodel = 0; imodel < render_data.dev_model_data->size; imodel++)
    {
        Model* model = &render_data.dev_model_data->pool[imodel];

        ray->points_transformed.orig = glm::vec3(model->world_to_model * glm::vec4(ray->points_base.orig, 1.0f));
        ray->points_transformed.end = glm::vec3(model->world_to_model * glm::vec4(ray->points_base.end, 1.0f));

        computeRayGridIntersection(render_data, iray, imodel);
    }
}

__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere( glm::vec3 normal, thrust::default_random_engine& rng) 
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

void shadeRay(int iter, RenderData render_data)
{
    for (int iray = 0; iray < render_data.dev_ray_data->size; iray++)
    {
        Ray* ray = &render_data.dev_ray_data->pool[iray];

        if (ray->hit_info.t < FLOAT_MAX)
        {
            glm::vec3 dir = glm::normalize(ray->points_base.end - ray->points_base.orig);
            glm::vec3 intersection_pt = ray->points_base.orig + dir * ray->hit_info.t;

            if (ray->meta_data.remainingBounces > 0)
            {
                if (ray->hit_info.impact_mat.material_type == Material::MaterialType::DIFFUSE)
                {
                    ray->color *= ray->hit_info.impact_mat.color;
                    thrust::default_random_engine rng = makeSeededRandomEngine(iter, iray, 0);
                    glm::vec3 scattered_ray_dir = calculateRandomDirectionInHemisphere(ray->hit_info.impact_normal, rng);
                    ray->points_base.orig = intersection_pt + 0.5f * ray->hit_info.impact_normal;
                    ray->points_base.end = ray->points_base.orig + scattered_ray_dir * 10.0f;
                }
                else if (ray->hit_info.impact_mat.material_type == Material::MaterialType::EMISSIVE)
                {
                    ray->color *= ray->hit_info.impact_mat.color;
                    ray->meta_data.remainingBounces = 0;
                }
            }
            ray->hit_info.t = FLOAT_MAX;
        }
        else
        {
            ray->meta_data.remainingBounces = 0;
            ray->color *= glm::vec3(0.001f, 0.001f, 0.001f);
            ray->hit_info.t = FLOAT_MAX;
        }
        ray->meta_data.remainingBounces--;
    }
}


__global__ void shadeRay_kernel(int nrays, int iter, RenderData render_data)
{
    int iray = threadIdx.x + blockDim.x * blockIdx.x;

    if (iray >= nrays) return;

    Ray* ray = &render_data.dev_ray_data->pool[iray];

    if (ray->hit_info.t < FLOAT_MAX)
    {
        glm::vec3 dir = glm::normalize(ray->points_base.end - ray->points_base.orig);
        glm::vec3 intersection_pt = ray->points_base.orig + dir * ray->hit_info.t;

        if (ray->meta_data.remainingBounces > 0)
        {
            if (ray->hit_info.impact_mat.material_type == Material::MaterialType::DIFFUSE)
            {
                ray->color *= ray->hit_info.impact_mat.color;
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, iray, 0);
                glm::vec3 scattered_ray_dir = calculateRandomDirectionInHemisphere(ray->hit_info.impact_normal, rng);
                ray->points_base.orig = intersection_pt + 0.1f * ray->hit_info.impact_normal;
                ray->points_base.end = ray->points_base.orig + scattered_ray_dir * 10.0f;
            }
            else if (ray->hit_info.impact_mat.material_type == Material::MaterialType::EMISSIVE)
            {
                ray->meta_data.remainingBounces = 0;
                ray->color *= ray->hit_info.impact_mat.color;
                ray->hit_info.t = FLOAT_MAX;
                return;
            }
        }
        ray->hit_info.t = FLOAT_MAX;
    }
    else
    {
        ray->meta_data.remainingBounces = 0;
        ray->color *= glm::vec3(0.01f, 0.01f, 0.01f);
        ray->hit_info.t = FLOAT_MAX;
        return;
    }
    ray->meta_data.remainingBounces--;
}

void computeRaySceneIntersection(RenderData render_data)
{
    for (int iray = 0; iray < render_data.dev_ray_data->size; iray++)
    {
        Ray* ray = &render_data.dev_ray_data->pool[iray];
        for (int imodel = 0; imodel < render_data.dev_model_data->size; imodel++)
        {
            if (iray != (220 + (RESOLUTION_Y - 450 - 1) * RESOLUTION_X)) continue;
            Model* model = &render_data.dev_model_data->pool[imodel];

            ray->points_transformed.orig = glm::vec3(model->world_to_model * glm::vec4(ray->points_base.orig, 1.0f));
            ray->points_transformed.end = glm::vec3(model->world_to_model * glm::vec4(ray->points_base.end, 1.0f));

            computeRayGridIntersection(render_data, iray, imodel);
            #ifdef ENABLE_VISUALIZER
                render_data.visualizer_data.rays.push_back(iray);
            #endif
           
        }
    }
}

__global__ void finalGather(RenderData render_data)
{
    int iray = threadIdx.x + blockDim.x * blockIdx.x;

    if (iray < render_data.dev_ray_data->size)
    {
        Ray *ray = &render_data.dev_ray_data->pool[iray];
        render_data.dev_image_data->pool[ray->meta_data.ipixel].color += ray->color;
    }
}
struct hasTerminated
{
    __host__ __device__
        bool operator()(const int& x)
    {
        return x == 1;
    }
};

__global__ void compactStencil_kernel(int nrays, Ray* dev_ray_data, int* dev_stencil)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nrays)
    {
        if (dev_ray_data[index].meta_data.remainingBounces <= 0)
        {
            dev_stencil[index] = 0;
            return;
        }
        dev_stencil[index] = 1;
    }
}

__global__ void generateRays_kernel(int nrays, RenderData render_data)
{
    int iray = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iray < nrays)
    {
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
        render_data.dev_ray_data->pool[iray].points_base.end = pix_pos;
        render_data.dev_ray_data->pool[iray].hit_info.t = FLOAT_MAX;
        render_data.dev_ray_data->pool[iray].color = glm::vec3(1.0f, 1.0f, 1.0f);
        render_data.dev_ray_data->pool[iray].meta_data.remainingBounces = 5;
        render_data.dev_ray_data->pool[iray].meta_data.ipixel = iray;
    }
}

__global__ void initImage_kernel(int nrays, RenderData render_data)
{
    int iray = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (iray < nrays)
    {
        render_data.dev_image_data->pool[iray].color = glm::vec3(0.0f, 0.0f, 0.0f);
    }
}

void Renderer::renderLoop()
{
    int count = 0;
    cudaError_t err;

    dim3 threads(32);
    dim3 blocks = (ceil(render_data.dev_ray_data->size / 32));

    auto start_time = std::chrono::high_resolution_clock::now();

    initImage_kernel << <blocks, threads >> > (render_data.dev_ray_data->size, render_data);
    err = cudaDeviceSynchronize();
   
    for (int iter = 0; iter < ITER; iter++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        bool iterationComplete = false;
        int nrays = render_data.dev_ray_data->size;

        generateRays_kernel << <blocks, threads >> > (nrays, render_data);
        err = cudaDeviceSynchronize();

        while (!iterationComplete)
        {
            blocks = (ceil(render_data.dev_ray_data->size / 32));

            computeRaySceneIntersection_kernel << <blocks, threads >> > (nrays, render_data);
            err = cudaDeviceSynchronize();

            shadeRay_kernel << <blocks, threads >> > (nrays, iter, render_data);
            err = cudaDeviceSynchronize();

            compactStencil_kernel << <blocks, threads >> > (nrays, render_data.dev_ray_data->pool, render_data.dev_stencil->pool);
            err = cudaDeviceSynchronize();

            Ray* itr = thrust::stable_partition(thrust::device, render_data.dev_ray_data->pool, render_data.dev_ray_data->pool + nrays, render_data.dev_stencil->pool, hasTerminated());
            int n = itr - render_data.dev_ray_data->pool;
            nrays = n;

            if (nrays == 0)
            {
                iterationComplete = true;
            }
        }
        finalGather << <blocks, threads >> > (render_data);
        err = cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::cout << "Iteration "<<iter+1<<": " << duration.count() << " microseconds" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Full run: " << duration.count() << " microseconds" << std::endl;
}