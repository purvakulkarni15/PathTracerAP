#include "Renderer.h"

#define EPSILON 0.1f

void Renderer::renderImage()
{
    computeRaySceneIntersection();
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

    // Write the BMP header
    outFile.write(bmpHeader, sizeof(bmpHeader));

    // Write the pixel data (24 bits per pixel, RGB)
    for (int y = 0; y < RESOLUTION_Y; ++y) {
        for (int x = 0; x < RESOLUTION_X; ++x)
        {
            char pixel[] = { dev_rays->pool[x + y * RESOLUTION_X].color.x, dev_rays->pool[x + y * RESOLUTION_X].color.y, dev_rays->pool[x + y * RESOLUTION_X].color.z };
            outFile.write(pixel, sizeof(pixel));
        }
    }

    // Update file size and image size in the header
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
    dev_models = GPUMemoryPool<Model>::getInstance();
    dev_models->allocate(scene.models);

    dev_meshes = GPUMemoryPool<Mesh>::getInstance();
    dev_meshes->allocate(scene.meshes);

	dev_vertexDataArr = GPUMemoryPool<VertexData>::getInstance();
	dev_vertexDataArr->allocate(scene.vertex_data_pool);

	dev_triangles = GPUMemoryPool<Triangle>::getInstance();
	dev_triangles->allocate(scene.triangles);

    dev_grids = GPUMemoryPool<Grid>::getInstance();
    dev_grids->allocate(scene.grids);

    dev_voxels = GPUMemoryPool<Range>::getInstance();
    dev_voxels->allocate(scene.voxels);

    dev_perVoxelDataPool = GPUMemoryPool<int>::getInstance();
    dev_perVoxelDataPool->allocate(scene.per_voxel_data_pool);
}

void Renderer::addRays(vector<Ray> rays)
{
    dev_rays = GPUMemoryPool<Ray>::getInstance();
    dev_rays->allocate(rays);
}

void Renderer::computeRaySceneIntersection()
{
    for (int i = 0; i < dev_rays->size; i++)
    {
        for (int j = 0; j < dev_models->size; j++)
        {
            dev_rays->pool[i].orig_transformed = glm::vec3(dev_models->pool[j].world_to_model * glm::vec4(dev_rays->pool[i].orig, 1.0f));
            dev_rays->pool[i].end_transformed = glm::vec3(dev_models->pool[j].world_to_model * glm::vec4(dev_rays->pool[i].end, 1.0f));

            int grid_index = dev_models->pool[j].grid_index;
            RenderData render_data;
            render_data.mat = dev_models->pool[j].mat;
            computeRayGridIntersection(dev_rays->pool[i], dev_grids->pool[grid_index], render_data);
        }
    }
}

bool Renderer::computeRayBoundingBoxIntersection(Ray& ray, glm::vec3 bounding_box[2])
{
    glm::vec3 dir = glm::normalize(ray.end_transformed - ray.orig_transformed);

    float inv_x_dir = 1 / dir.x;
    float inv_y_dir = 1 / dir.y;
    float inv_z_dir = 1 / dir.z;

    float t1 = (bounding_box[0].x - ray.orig_transformed.x) * inv_x_dir;
    float t2 = (bounding_box[1].x - ray.orig_transformed.x) * inv_x_dir;
    float t3 = (bounding_box[0].y - ray.orig_transformed.y) * inv_y_dir;
    float t4 = (bounding_box[1].y - ray.orig_transformed.y) * inv_y_dir;
    float t5 = (bounding_box[0].z - ray.orig_transformed.z) * inv_z_dir;
    float t6 = (bounding_box[1].z - ray.orig_transformed.z) * inv_z_dir;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0)
    {
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        return false;
    }

    ray.t = tmin;
    return true;
}

bool Renderer::computeRayTriangleIntersection(Triangle tri, Ray& ray, RenderData& render_data)
{
    VertexData vert1 = dev_vertexDataArr->pool[tri.indices[0]];
    VertexData vert2 = dev_vertexDataArr->pool[tri.indices[1]];
    VertexData vert3 = dev_vertexDataArr->pool[tri.indices[2]];

    glm::vec3 edge1 = vert2.vertex - vert1.vertex;
    glm::vec3 edge2 = vert3.vertex - vert1.vertex;

    glm::vec3 normal = glm::normalize((vert1.normal + vert2.normal + vert3.normal)/3.0f);//glm::normalize(glm::cross(edge1, edge2));

    glm::vec3 dir = glm::normalize(ray.end_transformed - ray.orig_transformed);

    glm::vec3 h = glm::cross(dir, edge2);
    float a = glm::dot(edge1, h);

    if (a > -EPSILON && a < EPSILON)
        return false;

    float f = 1.0f / a;
    glm::vec3 s = ray.orig_transformed - vert1.vertex;
    float u = f * glm::dot(s, h);

    if (u < -EPSILON || u > 1.0f + EPSILON)
        return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(dir, q);

    if (v < -EPSILON || u + v > 1.0f + EPSILON)
        return false;

    float t = f * glm::dot(edge2, q);
    if (ray.t > t)
    {
        ray.t = t;
        glm::vec3 intersection = t * dir + ray.orig_transformed;
        glm::vec3 light = glm::normalize(ray.orig_transformed - intersection);
        float c = std::max(glm::dot(light, normal), 0.0f);
        ray.color = c * render_data.mat.color;
    }
    return t > EPSILON;
}

bool Renderer::computeRayVoxelIntersection(Ray& ray, Range& voxel, RenderData& render_data)
{
    bool isIntersect = false;

    for (int i = voxel.start_index; i < voxel.end_index; i++)
    {
        Triangle tri = dev_triangles->pool[dev_perVoxelDataPool->pool[i]];
        if (computeRayTriangleIntersection(tri, ray, render_data)) isIntersect = true;
    }
    return isIntersect;
}

bool Renderer::computeRayGridIntersection(Ray& ray, Grid& grid, RenderData& render_data)
{
    glm::vec3 dir = glm::normalize(ray.end_transformed - ray.orig_transformed);

    int nVoxel = GRID_X * GRID_Y * GRID_Z;
    glm::vec3* bounding_box = dev_meshes->pool[grid.mesh_index].bounding_box;
    
    if (computeRayBoundingBoxIntersection(ray, bounding_box))
    {
        glm::vec3 currModelPos = ray.orig_transformed + dir * ray.t;

        ray.t = std::numeric_limits<float>::max();

        float x_width = bounding_box[1].x - bounding_box[0].x;
        float y_width = bounding_box[1].y - bounding_box[0].y;
        float z_width = bounding_box[1].z - bounding_box[0].z;

        float x_cell_width = x_width / GRID_X;
        float y_cell_width = y_width / GRID_Y;
        float z_cell_width = z_width / GRID_Z;

        if ((currModelPos.x - bounding_box[0].x) < -EPSILON)
        {
            ray.color = glm::vec3(0, 200, 0);
            return false;
        }
        if ((currModelPos.y - bounding_box[0].y) < -EPSILON)
        {
            ray.color = glm::vec3(200, 0, 0);
            return false;
        }
        if ((currModelPos.z - bounding_box[0].z) < -EPSILON)
        {
            ray.color = glm::vec3(0, 0, 200);
            return false;
        }

        int x_index = glm::floor((currModelPos.x - bounding_box[0].x) / x_cell_width);
        int y_index = glm::floor((currModelPos.y - bounding_box[0].y) / y_cell_width);
        int z_index = glm::floor((currModelPos.z - bounding_box[0].z) / z_cell_width);

        x_index = min(x_index, GRID_X - 1);
        y_index = min(y_index, GRID_Y - 1);
        z_index = min(z_index, GRID_Z - 1);

        x_index = max(x_index, 0);
        y_index = max(y_index, 0);
        z_index = max(z_index, 0);

        float fmax_val = std::numeric_limits<float>::max();
        glm::vec3 tMax = glm::vec3(fmax_val, fmax_val, fmax_val);
        glm::vec3 delta = glm::vec3(fmax_val, fmax_val, fmax_val);

        int step_x = 1, step_y = 1, step_z = 1;
        int out_x = GRID_X, out_y = GRID_Y, out_z = GRID_Z;

        float nextPosX = bounding_box[0].x + (x_index + 1) * x_cell_width;
        float prevPosX = bounding_box[0].x + (x_index)*x_cell_width;

        if (dir.x > 0.0f)
        {
            delta.x = x_cell_width / dir.x;
            tMax.x = (nextPosX - currModelPos.x) / dir.x;
            step_x = 1;
            out_x = GRID_X;
        }
        else if (dir.x < 0.0f)
        {
            delta.x = -1.0f * x_cell_width / dir.x;
            tMax.x = (prevPosX - currModelPos.x) / dir.x;
            step_x = -1;
            out_x = -1;
        }

        float nextPosY = bounding_box[0].y + (y_index + 1) * y_cell_width;
        float prevPosY = bounding_box[0].y + (y_index)*y_cell_width;

        if (dir.y > 0.0f)
        {
            delta.y = y_cell_width / dir.y;
            tMax.y = (nextPosY - currModelPos.y) / dir.y;
            step_y = 1;
            out_y = GRID_Y;
        }
        else if (dir.y < 0.0f)
        {
            delta.y = -1.0f * y_cell_width / dir.y;
            tMax.y = (prevPosY - currModelPos.y) / dir.y;
            step_y = -1;
            out_y = -1;
        }

        float nextPosZ = bounding_box[0].z + (z_index + 1) * z_cell_width;
        float prevPosZ = bounding_box[0].z + (z_index)*z_cell_width;

        if (dir.z > 0.0f)
        {
            delta.z = z_cell_width / dir.z;
            tMax.z = (nextPosZ - currModelPos.z) / dir.z;
            step_z = 1;
            out_z = GRID_Z;
        }
        else if (dir.z < 0.0f)
        {
            delta.z = -1.0f * z_cell_width / dir.z;
            tMax.z = (prevPosZ - currModelPos.z) / dir.z;
            step_z = -1;
            out_z = -1;
        }

        while (1)
        {
            int index = x_index + y_index * GRID_X + z_index * GRID_X * GRID_Y;
            if (computeRayVoxelIntersection(ray, dev_voxels->pool[index], render_data))
            {
                return true;
            }

            if (tMax.x < tMax.y && tMax.x < tMax.z)
            {
                x_index += step_x;
                if (x_index == out_x)
                {
                    return false;
                }
                tMax.x += delta.x;
            }
            else if (tMax.y < tMax.z)
            {
                y_index += step_y;
                if (y_index == out_y)
                {
                    return false;
                }
                tMax.y += delta.y;
            }
            else
            {
                z_index += step_z;
                if (z_index == out_z)
                {
                    return false;
                }
                tMax.z += delta.z;
            }
        }
    }
}