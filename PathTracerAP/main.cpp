#include "PathTracing.h"
#include <algorithm>

#define EPSILON 0.000001f

int main() 
{
	Camera::generateRays();
	GPU::allocateMemoryForRays(Camera::rays);

	Scene::Mesh mesh;
	mesh.loadMesh("dragon.obj");
	vector<Scene::Mesh> meshes;
	meshes.push_back(mesh);
	GPU::allocateMemoryForMeshes(meshes);
	UniformGrid::addUniformGrid(GPU::dev_triangles, meshes[0].st, meshes[0].ed, meshes[0].bounding_box);

	GPU::allocateMemoryForGrids(UniformGrid::grids);
	GPU::computeRayMeshIntersection();

	Camera::generateImage();

}


bool Triangle::intersectRay(Ray& ray)
{
    glm::vec3 vert1 = *(pVerts[0]);
    glm::vec3 vert2 = *(pVerts[1]);
    glm::vec3 vert3 = *(pVerts[2]);

    glm::vec3 edge1 = vert2 - vert1;
    glm::vec3 edge2 = vert3 - vert1;

    glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

    glm::vec3 h = glm::cross(ray.dir, edge2);
    float a = glm::dot(edge1, h);

    if (a > -EPSILON && a < EPSILON)
        return false;

    float f = 1.0f / a;
    glm::vec3 s = ray.orig - vert1;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.dir, q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = f * glm::dot(edge2, q);
    if (ray.t > t)
    {
        ray.t = t;
        glm::vec3 intersection = t * ray.dir + ray.orig;
        glm::vec3 light = glm::normalize(ray.orig - intersection);
        float c = std::max(glm::dot(light, normal), 0.0f);
        ray.color = c*glm::vec3(0, 0, 200);
    }
    return t > EPSILON;
}

void UniformGrid::addUniformGrid(Triangle* dev_triangles, int st, int ed, glm::vec3 bounding_box[2])
{
  
    Grid_CPU grid;
    grid.nTriangles = 0;
    grid.bounding_box[0] = bounding_box[0];
    grid.bounding_box[1] = bounding_box[1];

    float x_width = grid.bounding_box[1].x - grid.bounding_box[0].x;
    float y_width = grid.bounding_box[1].y - grid.bounding_box[0].y;
    float z_width = grid.bounding_box[1].z - grid.bounding_box[0].z;

    float x_cell_width = x_width / GRID_X;
    float y_cell_width = y_width / GRID_Y;
    float z_cell_width = z_width / GRID_Z;

    grid.voxels.resize(GRID_X * GRID_Y * GRID_Z);

    for (int t = st; t <= ed; t++)
    {
        glm::vec3 v1 = *(dev_triangles[t].pVerts[0]);
        glm::vec3 v2 = *(dev_triangles[t].pVerts[1]);
        glm::vec3 v3 = *(dev_triangles[t].pVerts[2]);

        float max_x = v1.x > v2.x ? v1.x : v2.x;
        max_x = max_x > v3.x ? max_x : v3.x;

        float max_y = v1.y > v2.y ? v1.y : v2.y;
        max_y = max_y > v3.y ? max_y : v3.y;

        float max_z = v1.z > v2.z ? v1.z : v2.z;
        max_z = max_z > v3.z ? max_z : v3.z;

        float min_x = v1.x < v2.x ? v1.x : v2.x;
        min_x = min_x < v3.x ? min_x : v3.x;

        float min_y = v1.y < v2.y ? v1.y : v2.y;
        min_y = min_y < v3.y ? min_y : v3.y;

        float min_z = v1.z < v2.z ? v1.z : v2.z;
        min_z = min_z > v3.z ? min_z : v3.z;

        int cell_x_st = floor(abs(bounding_box[0].x - min_x) / x_cell_width);
        int cell_x_ed = floor(abs(bounding_box[0].x - max_x) / x_cell_width);

        int cell_y_st = floor(abs(bounding_box[0].y - min_y) / y_cell_width);
        int cell_y_ed = floor(abs(bounding_box[0].y - max_y) / y_cell_width);

        int cell_z_st = floor(abs(bounding_box[0].z - min_z) / z_cell_width);
        int cell_z_ed = floor(abs(bounding_box[0].z - max_z) / z_cell_width);

        cell_x_ed = min(cell_x_ed, GRID_X - 1);
        cell_y_ed = min(cell_y_ed, GRID_Y - 1);
        cell_z_ed = min(cell_z_ed, GRID_Z - 1);

        cell_x_st = max(cell_x_st, 0);
        cell_y_st = max(cell_y_st, 0);
        cell_z_st = max(cell_z_st, 0);

        for (int z = cell_z_st; z <= cell_z_ed; z++)
        {
            for (int y = cell_y_st; y <= cell_y_ed; y++)
            {
                for (int x = cell_x_st; x <= cell_x_ed; x++)
                {
                    int index = x + y * GRID_X + GRID_X * GRID_Y * z;
                    grid.voxels[index].triangle_indices.push_back(t);
                    grid.nTriangles++;
                }
            }
        }
    }

    grids.push_back(grid);
}

bool UniformGrid::intersectBoundingBox(Ray& ray, glm::vec3 bounding_box[2])
{
    float inv_x_dir = 1 / ray.dir.x;
    float inv_y_dir = 1 / ray.dir.y;
    float inv_z_dir = 1 / ray.dir.z;

    float t1 = (bounding_box[0].x - ray.orig.x) * inv_x_dir;
    float t2 = (bounding_box[1].x - ray.orig.x) * inv_x_dir;
    float t3 = (bounding_box[0].y - ray.orig.y) * inv_y_dir;
    float t4 = (bounding_box[1].y - ray.orig.y) * inv_y_dir;
    float t5 = (bounding_box[0].z - ray.orig.z) * inv_z_dir;
    float t6 = (bounding_box[1].z - ray.orig.z) * inv_z_dir;

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
    //ray.color = glm::vec3(0, 0, 200);
    return true;
}

bool UniformGrid::intersectVoxel(Ray& ray, Voxel_GPU& voxel)
{
    bool isIntersect = false;
    int st = voxel.start_index;
    int ed = voxel.end_index;
    for (int i = st; i < ed; i++)
    {
        Triangle tri = GPU::dev_triangles[GPU::dev_triangleIndices[i]];
        if (tri.intersectRay(ray)) isIntersect = true;
    }
    return isIntersect;
}

bool UniformGrid::intersectGrid(Ray& ray, Grid_GPU& grid)
{
    int nVoxel = GRID_X * GRID_Y * GRID_Z;
    if (intersectBoundingBox(ray, grid.bounding_box))
    {
        //return true;
        glm::vec3 currPos = ray.orig + ray.dir * ray.t;
        ray.t = std::numeric_limits<float>::max();
        float x_width = grid.bounding_box[1].x - grid.bounding_box[0].x;
        float y_width = grid.bounding_box[1].y - grid.bounding_box[0].y;
        float z_width = grid.bounding_box[1].z - grid.bounding_box[0].z;

        float x_cell_width = x_width / GRID_X;
        float y_cell_width = y_width / GRID_Y;
        float z_cell_width = z_width / GRID_Z;

        if ((currPos.x - grid.bounding_box[0].x) < -EPSILON)
        {
            return false;
        }
        if ((currPos.y - grid.bounding_box[0].y) < -EPSILON)
        {
            return false;
        }
        if ((currPos.z - grid.bounding_box[0].z) < -EPSILON)
        {
            return false;
        }

        int x_index = glm::floor((currPos.x - grid.bounding_box[0].x) / x_cell_width);
        int y_index = glm::floor((currPos.y - grid.bounding_box[0].y) / y_cell_width);
        int z_index = glm::floor((currPos.z - grid.bounding_box[0].z) / z_cell_width);

        x_index = min(x_index, GRID_X - 1);
        y_index = min(y_index, GRID_Y - 1);
        z_index = min(z_index, GRID_Z - 1);

        glm::vec3 tMax = glm::vec3(99999.0f, 99999.0f, 99999.0f);
        glm::vec3 delta = glm::vec3(99999.0f, 99999.0f, 99999.0f);

        int step_x = 1, step_y = 1, step_z = 1;
        int out_x = GRID_X, out_y = GRID_Y, out_z = GRID_Z;

        float nextPosX = grid.bounding_box[0].x + (x_index + 1) * x_cell_width;
        float prevPosX = grid.bounding_box[0].x + (x_index)*x_cell_width;

        if (ray.dir.x > 0.0f)
        {
            delta.x = x_cell_width / ray.dir.x;
            tMax.x = (nextPosX - currPos.x) / ray.dir.x;
            step_x = 1;
            out_x = GRID_X;
        }
        else if (ray.dir.x < 0.0f)
        {
            delta.x = -1.0f * x_cell_width / ray.dir.x;
            tMax.x = (prevPosX - currPos.x) / ray.dir.x;
            step_x = -1;
            out_x = -1;
        }

        float nextPosY = grid.bounding_box[0].y + (y_index + 1) * y_cell_width;
        float prevPosY = grid.bounding_box[0].y + (y_index)*y_cell_width;

        if (ray.dir.y > 0.0f)
        {
            delta.y = y_cell_width / ray.dir.y;
            tMax.y = (nextPosY - currPos.y) / ray.dir.y;
            step_y = 1;
            out_y = GRID_Y;
        }
        else if (ray.dir.y < 0.0f)
        {
            delta.y = -1.0f * y_cell_width / ray.dir.y;
            tMax.y = (prevPosY - currPos.y) / ray.dir.y;
            step_y = -1;
            out_y = -1;
        }

        float nextPosZ = grid.bounding_box[0].z + (z_index + 1) * z_cell_width;
        float prevPosZ = grid.bounding_box[0].z + (z_index)*z_cell_width;

        if (ray.dir.z > 0.0f)
        {
            delta.z = z_cell_width / ray.dir.z;
            tMax.z = (nextPosZ - currPos.z) / ray.dir.z;
            step_z = 1;
            out_z = GRID_Z;
        }
        else if (ray.dir.z < 0.0f)
        {
            delta.z = -1.0f * z_cell_width / ray.dir.z;
            tMax.z = (prevPosZ - currPos.z) / ray.dir.z;
            step_z = -1;
            out_z = -1;
        }

        while (1)
        {
            int index = x_index + y_index * GRID_X + z_index * GRID_X * GRID_Y;
            if (intersectVoxel(ray, GPU::dev_voxels[index]))
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


void Camera::generateRays()
{
    glm::vec3 camera_orig = glm::vec3(0, 0, 17);
    rays.resize(RESOLUTION_X * RESOLUTION_Y);

    float world_x = -10;
    float step_x = 20.0 / RESOLUTION_X;
    float world_y = -8;
    float step_y = 16.0 / RESOLUTION_Y;
    float world_z = 15.0;


    for (int y = 0; y < RESOLUTION_Y; ++y)
    {
        world_x = -10;
        for (int x = 0; x < RESOLUTION_X; ++x)
        {
            int index = y * RESOLUTION_X + x;
            glm::vec3 pix_pos = glm::vec3(world_x, world_y, world_z);
            
            Camera::rays[index].orig = camera_orig;
            Camera::rays[index].dir = glm::normalize(pix_pos - camera_orig);
            Camera::rays[index].t = std::numeric_limits<float>::max();
            Camera::rays[index].color = glm::vec3(200, 200, 200);
            world_x += step_x;
        }
        world_y += step_y;
    }


}

void GPU::allocateMemoryForMeshes(vector<Scene::Mesh>& meshes)
{
    nVertices = 0;
    nTriangles = 0;

    for (int i = 0; i < meshes.size(); i++)
    {
        nVertices += meshes[i].vertices.size();
        nTriangles += meshes[i].indices.size() / 3;
    }

    dev_vertices = (glm::vec3*)malloc(sizeof(glm::vec3) * nVertices);
    dev_normals = (glm::vec3*)malloc(sizeof(glm::vec3) * nVertices);
    dev_uvs = (glm::vec2*)malloc(sizeof(glm::vec2) * nVertices);
    dev_triangles = (Triangle*)malloc(sizeof(Triangle) * nTriangles);

    int v_index = 0;
    int t_index = 0;

    for (int i = 0; i < meshes.size(); i++)
    {
        int offset = v_index;
        for (int v = 0; v < meshes[i].vertices.size(); v++)
        {
            dev_vertices[v_index].x = meshes[i].vertices[v].x;
            dev_vertices[v_index].y = meshes[i].vertices[v].y;
            dev_vertices[v_index].z = meshes[i].vertices[v].z;
            /*dev_normals[v_index] = meshes[i].normals[v];
            dev_uvs[v_index] = meshes[i].uvs[v];*/
            v_index++;
        }
        
        int nT = meshes[i].indices.size() / 3;
        meshes[i].st = t_index;
        for (int t = 0; t < nT; t++)
        {
            int v1_ind = meshes[i].indices[t * 3];
            int v2_ind = meshes[i].indices[t * 3 + 1];
            int v3_ind = meshes[i].indices[t * 3 + 2];

            dev_triangles[t_index].pVerts[0] = &dev_vertices[v1_ind + offset];
            dev_triangles[t_index].pVerts[1] = &dev_vertices[v2_ind + offset];
            dev_triangles[t_index].pVerts[2] = &dev_vertices[v3_ind + offset];
            t_index++;
        }
        meshes[i].ed = t_index - 1;
    }
}

void GPU::allocateMemoryForGrids(vector<Grid_CPU>& grids)
{
    GPU::nGrid = grids.size();
    int nGrid = grids.size();
    int nVoxel = GRID_X * GRID_Y * GRID_Z;

    dev_grids = (Grid_GPU*)malloc(sizeof(Grid_GPU) * nGrid);
    dev_voxels = (Voxel_GPU*)malloc(sizeof(Voxel_GPU) * nGrid * nVoxel);

    int nTriangles = 0;

    int t_index = 0;
    int v_index = 0;

    for (int i = 0; i < grids.size(); i++)
    {
        dev_grids[i].start_index = v_index;
        dev_grids[i].end_index = v_index + nVoxel - 1;

        nTriangles += grids[i].nTriangles;
        dev_grids[i].bounding_box[0] = grids[i].bounding_box[0];
        dev_grids[i].bounding_box[1] = grids[i].bounding_box[1];

        //dev_grids[i].bounding_box[0] = glm::vec3(-10.0, -10.0, -10.0);
        //dev_grids[i].bounding_box[1] = glm::vec3(10.0, 10.0, 10.0);

        v_index += nVoxel;
    }
    dev_triangleIndices = (int*)malloc(sizeof(int) * nTriangles);

    v_index = 0;

    for (int i = 0; i < grids.size(); i++)
    {
        for (int j = 0; j < grids[i].voxels.size(); j++)
        {
            dev_voxels[v_index].start_index = t_index;
            if (t_index < 0)
            {
                cout << "AAAHHHHH" << endl;
            }
            for (int k = 0; k < grids[i].voxels[j].triangle_indices.size(); k++)
            {
                dev_triangleIndices[t_index] = grids[i].voxels[j].triangle_indices[k];
                t_index++;
            }
            dev_voxels[v_index].end_index = t_index;
            v_index++;
        }
    }
    cout << "tindex" << t_index;
}

void GPU::computeRayMeshIntersection()
{
    for (int i = 0; i < nRay; i++)
    {
        if (i == 231+500*(399-185))
        {
            cout << "jj" << endl;
        }
        for (int j = 0; j < nGrid; j++)
        {
            UniformGrid::intersectGrid(rays[i], dev_grids[j]);
        }
    }
}

void GPU::allocateMemoryForRays(vector<Ray>& rays_in)
{
    nRay = rays_in.size();
    GPU::rays = (Ray*)malloc(sizeof(Ray) * nRay);

    for (int i = 0; i < nRay; i++)
    {
        GPU::rays[i] = rays_in[i];
    }
}

void Camera::generateImage()
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

    std::ofstream outFile("RT.bmp", std::ios::binary);

    // Write the BMP header
    outFile.write(bmpHeader, sizeof(bmpHeader));

    // Write the pixel data (24 bits per pixel, RGB)
    for (int y = 0; y < RESOLUTION_Y; ++y) {
        for (int x = 0; x < RESOLUTION_X; ++x)
        {
            char pixel[] = { GPU::rays[x + y * RESOLUTION_X].color.x, GPU::rays[x + y * RESOLUTION_X].color.y, GPU::rays[x + y * RESOLUTION_X].color.z };
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

/*bool Scene::Sphere::intersectRay(glm::vec3 ray_orig, glm::vec3 ray_dir, float& out_dist)
{
    glm::vec3 oc = ray_orig - center;
    float a = glm::dot(ray_dir, ray_dir);
    float b = 2.0f * glm::dot(oc, ray_dir);
    float c = glm::dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant >= 0)
    {
        float t1 = (-b - glm::sqrt(discriminant)) / (2.0f * a);
        float t2 = (-b + glm::sqrt(discriminant)) / (2.0f * a);

        if (t1 > 0 || t2 > 0) {
            out_dist = (t1 < t2) ? t1 : t2;
            return true;
        }
    }

    return false;
}*/

void Scene::Mesh::loadMesh(string path)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        cerr << "Error loading mesh: " << importer.GetErrorString() << endl;
        return;
    }

    processNode(scene->mRootNode, scene);
}

void Scene::Mesh::processNode(aiNode* node, const aiScene* scene)
{
    // Process all the node's meshes (if any)
    for (unsigned int i = 0; i < node->mNumMeshes; ++i)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, scene);
    }

    // Recursively process child nodes
    for (unsigned int i = 0; i < node->mNumChildren; ++i)
    {
        processNode(node->mChildren[i], scene);
    }
}

void Scene::Mesh::processMesh(aiMesh* mesh, const aiScene* scene)
{
    bounding_box[0].x = std::numeric_limits<float>::max();
    bounding_box[0].y = std::numeric_limits<float>::max();
    bounding_box[0].z = std::numeric_limits<float>::max();
    bounding_box[1].x = std::numeric_limits<float>::min();
    bounding_box[1].y = std::numeric_limits<float>::min();
    bounding_box[1].z = std::numeric_limits<float>::min();

    for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
    {
        glm::vec3 vertex;
        vertex.x = mesh->mVertices[i].x * 100;
        bounding_box[0].x = bounding_box[0].x > vertex.x ? vertex.x : bounding_box[0].x;
        bounding_box[1].x = bounding_box[1].x < vertex.x ? vertex.x : bounding_box[1].x;
        vertex.y = mesh->mVertices[i].y * 100;
        bounding_box[0].y = bounding_box[0].y > vertex.y ? vertex.y : bounding_box[0].y;
        bounding_box[1].y = bounding_box[1].y < vertex.y ? vertex.y : bounding_box[1].y;
        vertex.z = mesh->mVertices[i].z * 100;
        bounding_box[0].z = bounding_box[0].z > vertex.z ? vertex.z : bounding_box[0].z;
        bounding_box[1].z = bounding_box[1].z < vertex.z ? vertex.z : bounding_box[1].z;
        vertices.push_back(vertex);

        /*glm::vec3 normal;
        normal.x = mesh->mNormals[i].x;
        normal.y = mesh->mNormals[i].y;
        normal.z = mesh->mNormals[i].z;
        normals.push_back(normal);

        if (mesh->mTextureCoords[0])
        {
            glm::vec2 texCoord;
            texCoord.x = mesh->mTextureCoords[0][i].x;
            texCoord.y = mesh->mTextureCoords[0][i].y;
            uvs.push_back(texCoord);
        }
        else
        {
            uvs.push_back(glm::vec2(0.0f, 0.0f));
        }*/
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; ++i)
    {
        aiFace face = mesh->mFaces[i];
        assert(face.mNumIndices == 3); // Assuming triangles

        glm::vec3* pVerts[3];
        for (unsigned int j = 0; j < 3; ++j)
        {
            indices.push_back(face.mIndices[j]);
        }
    }
}