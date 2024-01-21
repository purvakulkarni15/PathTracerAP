#pragma once
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

namespace Experiment
{
    __host__ __device__
        bool computeRayTriangleIntersection(RenderData& render_data, Ray* ray, IntersectionData* hit_info, int itriangle, Model* model)
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
        if (IS_LESS_THAN(u, 0.0f) || IS_MORE_THAN(u, 1.0f)) return false;

        glm::vec3 qvec = glm::cross(tvec, v0v1);
        float v = glm::dot(ray->transformed.dir, qvec) * invDet;
        if (IS_LESS_THAN(v, 0.0f) || IS_MORE_THAN(u + v, 1.0f)) return false;

        float t = glm::dot(v0v2, qvec) * invDet;

        if (IS_LESS_THAN(t, 0.0f)) return false;

        glm::vec3 normal = glm::cross(v0v1, v0v2);

        //if (det < 0) normal = glm::cross(v0v2, v0v1);
        //else normal = glm::cross(v0v1, v0v2);

        glm::vec3 model_coords_intersection = ray->transformed.orig + ray->transformed.dir * t;
        glm::vec3 world_coords_intersection = transformPosition(model_coords_intersection, model->model_to_world);
        float world_impact_distance = glm::length(world_coords_intersection - ray->base.orig);

        if (hit_info->impact_distance > world_impact_distance)
        {
            hit_info->impact_distance = world_impact_distance;
            hit_info->impact_normal = glm::normalize(transformNormal(normal, model->model_to_world));
            hit_info->impact_mat = model->mat;
        }

        return true;
    }

    __inline__ __host__ __device__
        bool computeRayVoxelIntersection(RenderData& render_data, Ray* ray, IntersectionData* hit_info, int ivoxel, Model* model)
    {
        Voxel* voxel = &render_data.dev_voxel_data->pool[ivoxel];

        bool isIntersect = false;

        if (voxel->entity_type == EntityType::TRIANGLE)
        {
            for (int i = voxel->entity_index_range.start_index; i < voxel->entity_index_range.end_index; i++)
            {
                int itriangle = render_data.dev_per_voxel_data->pool[i];

                if (computeRayTriangleIntersection(render_data, ray, hit_info, itriangle, model)) isIntersect = true;
            }
        }
        return isIntersect;
    }

    __global__
        void computeRayGridIntersection_kernel(int nrays, RenderData render_data, int imodel)
    {
        int iray = threadIdx.x + blockDim.x * blockIdx.x;
        if (iray >= nrays) return;
        
        Model* model = &render_data.dev_model_data->pool[imodel];
        Ray* ray = &render_data.dev_ray_data->pool[iray];
        IntersectionData* hit_info = &render_data.dev_intersection_data->pool[iray];
        Grid* grid = &render_data.dev_grid_data->pool[model->grid_index];
        float t_box = render_data.dev_dist_bounding_box->pool[iray];

        BoundingBox* bounding_box = &render_data.dev_mesh_data->pool[model->mesh_index].bounding_box;

        glm::vec3 grid_intersection_pt = ray->transformed.orig + ray->transformed.dir * t_box;

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
            if (computeRayVoxelIntersection(render_data, ray, hit_info, ivoxel, model))
            {
                ivoxel_cache = ivoxel_3d;
                is_intersect = true;
            }

            if (is_intersect && (ABS(ivoxel_cache.x - ivoxel_3d.x) > 2 || ABS(ivoxel_cache.y - ivoxel_3d.y) > 2 || ABS(ivoxel_cache.z - ivoxel_3d.z) > 2))
            {
                return;
            }

            if (tMax.x < tMax.y && tMax.x < tMax.z)
            {
                ivoxel_3d.x += step_x;
                if (ivoxel_3d.x == out_x || tMax.x >= FLOAT_MAX)
                {
                    return;
                }
                tMax.x += delta.x;
            }
            else if (tMax.y < tMax.z)
            {
                ivoxel_3d.y += step_y;
                if (ivoxel_3d.y == out_y || tMax.y >= FLOAT_MAX)
                {
                    return;
                }
                tMax.y += delta.y;
            }
            else
            {
                ivoxel_3d.z += step_z;
                if (ivoxel_3d.z == out_z || tMax.z >= FLOAT_MAX)
                {
                    return;
                }
                tMax.z += delta.z;
            }
        }
    }


    __global__
        void transformRayInModelSpace_kernel(int nrays, Ray* dev_ray_data, Model* model)
    {
        int iray = threadIdx.x + blockDim.x * blockIdx.x;
        if (iray >= nrays) return;

        Ray* ray = &dev_ray_data[iray];

        ray->transformed.orig = transformPosition(ray->base.orig, model->world_to_model);
        ray->transformed.dir = glm::normalize(transformDirection(ray->base.dir, model->world_to_model));
        ray->cache.inv_dir = glm::vec3(1 / ray->transformed.dir.x, 1 / ray->transformed.dir.y, 1 / ray->transformed.dir.z);
    }

    __global__
        void computeRayBoundingBoxIntersection_kernel(int nrays, Model* model, Mesh* mesh, Ray* dev_ray_data, IntersectionData* dev_intersection_data, int* dev_stencil_bounding_box_intersection, float* dev_t_bounding_box)
    {
        int iray = threadIdx.x + blockDim.x * blockIdx.x;
        if (iray >= nrays) return;

        Ray* ray = &dev_ray_data[iray];
        IntersectionData* hit_info = &dev_intersection_data[iray];
        BoundingBox* bounding_box = &mesh->bounding_box;

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
            dev_stencil_bounding_box_intersection[iray] = 0;
            dev_t_bounding_box[iray] = FLOAT_MAX;
            return;
        }

        glm::vec3 model_coords_intersection = ray->transformed.orig + ray->transformed.dir * tmin;
        glm::vec3 world_coords_intersection = transformPosition(model_coords_intersection, model->model_to_world);
        float world_impact_distance = glm::length(world_coords_intersection - ray->base.orig);

        if (hit_info->impact_distance < world_impact_distance)
        {
            dev_stencil_bounding_box_intersection[iray] = 0;
            dev_t_bounding_box[iray] = FLOAT_MAX;
        }

        dev_stencil_bounding_box_intersection[iray] = 1;
        dev_t_bounding_box[iray] = tmin;
    }

    struct hasTerminated_exp
    {
        __host__ __device__
            bool operator()(const int& x)
        {
            return x == 1;
        }
    };

    void computeRaySceneIntersection(int nr, RenderData render_data)
    {
        cudaError err;

        int nrays = render_data.dev_ray_data->size;
        dim3 threads(32);
        dim3 blocks = (ceil(nrays / 32));

        for (int i = 0; i < render_data.dev_model_data->size; i++)
        {
            nrays = nr;
            Model* model = &render_data.dev_model_data->pool[i];

            transformRayInModelSpace_kernel << <blocks, threads >> > (nrays, render_data.dev_ray_data->pool, model);
            err = cudaDeviceSynchronize();

            Mesh* mesh = &render_data.dev_mesh_data->pool[model->mesh_index];
            computeRayBoundingBoxIntersection_kernel<<<blocks, threads>>>(nrays, model, mesh, render_data.dev_ray_data->pool, render_data.dev_intersection_data->pool, render_data.dev_stencil_bounding_box_intersection->pool, render_data.dev_dist_bounding_box->pool);
            err = cudaDeviceSynchronize();

            //shorten the kernel size removing all the rays that do not intersect the model bounding box
            Ray* itr = thrust::stable_partition(thrust::device, render_data.dev_ray_data->pool, render_data.dev_ray_data->pool + nrays, render_data.dev_stencil_bounding_box_intersection->pool, hasTerminated_exp());
            IntersectionData* itr2 = thrust::stable_partition(thrust::device, render_data.dev_intersection_data->pool, render_data.dev_intersection_data->pool + nrays, render_data.dev_stencil_bounding_box_intersection->pool, hasTerminated_exp());
            float* itr3 = thrust::stable_partition(thrust::device, render_data.dev_dist_bounding_box->pool, render_data.dev_dist_bounding_box->pool + nrays, render_data.dev_stencil_bounding_box_intersection->pool, hasTerminated_exp());
            int n = itr - render_data.dev_ray_data->pool;
            nrays = n;

            //do actual grid traversal for the remaining.
            computeRayGridIntersection_kernel << <blocks, threads >> > (nrays, render_data, i);
            err = cudaDeviceSynchronize();
        }
    }
}