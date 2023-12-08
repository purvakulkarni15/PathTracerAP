#include "Scene.h"

Scene::Scene(string config)
{
    Mesh mesh;
    addMesh(config, mesh);
    meshes.push_back(mesh);
    createGrids();
}

void Scene::addMesh(string path, Mesh& mesh)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        cerr << "Error loading mesh: " << importer.GetErrorString() << endl;
        return;
    }

    //Initialize bounding box
    mesh.bounding_box[0].x = std::numeric_limits<float>::max();
    mesh.bounding_box[0].y = std::numeric_limits<float>::max();
    mesh.bounding_box[0].z = std::numeric_limits<float>::max();
    mesh.bounding_box[1].x = std::numeric_limits<float>::min();
    mesh.bounding_box[1].y = std::numeric_limits<float>::min();
    mesh.bounding_box[1].z = std::numeric_limits<float>::min();

    processNode(scene->mRootNode, mesh, scene);
}

void Scene::processNode(aiNode* node, Mesh& mesh, const aiScene* scene)
{
    // Process all the node's meshes (if any)
    for (unsigned int i = 0; i < node->mNumMeshes; ++i)
    {
        aiMesh* ai_mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(ai_mesh, mesh, scene);
    }

    // Recursively process child nodes
    for (unsigned int i = 0; i < node->mNumChildren; ++i)
    {
        processNode(node->mChildren[i], mesh, scene);
    }
}

void updateBoundingBox(glm::vec3 *bounding_box, const glm::vec3 & vertex)
{
    bounding_box[0].x = bounding_box[0].x > vertex.x ? vertex.x : bounding_box[0].x;
    bounding_box[1].x = bounding_box[1].x < vertex.x ? vertex.x : bounding_box[1].x;
    bounding_box[0].y = bounding_box[0].y > vertex.y ? vertex.y : bounding_box[0].y;
    bounding_box[1].y = bounding_box[1].y < vertex.y ? vertex.y : bounding_box[1].y;
    bounding_box[0].z = bounding_box[0].z > vertex.z ? vertex.z : bounding_box[0].z;
    bounding_box[1].z = bounding_box[1].z < vertex.z ? vertex.z : bounding_box[1].z;
}

glm::vec3 getFromVector3D(const aiVector3D &vector3D)
{
    glm::vec3 vec3;
    vec3.x = vector3D.x;
    vec3.y = vector3D.y;
    vec3.z = vector3D.z;
    return vec3;
}

void Scene::processMesh(aiMesh* ai_mesh, Mesh& mesh, const aiScene* scene)
{
    mesh.vertexDataIndices.start_index = vertexDataArr.size();
    for (unsigned int i = 0; i < ai_mesh->mNumVertices; ++i)
    {
        VertexData vertexDataObj;
        vertexDataObj.vertex = getFromVector3D(ai_mesh->mVertices[i]);
        //vertexDataObj.normal = getFromVector3D(ai_mesh->mNormals[i]);
        vertexDataArr.push_back(vertexDataObj);

        updateBoundingBox(mesh.bounding_box, vertexDataObj.vertex);
    }
    mesh.vertexDataIndices.end_index = vertexDataArr.size();

    mesh.triangleDataIndices.start_index = triangles.size();
    for (unsigned int i = 0; i < ai_mesh->mNumFaces; ++i)
    {
        aiFace face = ai_mesh->mFaces[i];
        assert(face.mNumIndices == 3);

        Triangle tri;
        for (unsigned int j = 0; j < 3; ++j)
        {
            tri.indices[j] = mesh.vertexDataIndices.start_index + face.mIndices[j];
        }
        triangles.push_back(tri);
    }
    mesh.triangleDataIndices.end_index = triangles.size();
}

void getVoxelIndex(int& x_index_st, int& x_index_ed, int& y_index_st, int& y_index_ed, int& z_index_st, int& z_index_ed, const glm::vec3 grid_bounding_box[2], glm::vec3 triangle[3])
{
    float x_width = grid_bounding_box[1].x - grid_bounding_box[0].x;
    float y_width = grid_bounding_box[1].y - grid_bounding_box[0].y;
    float z_width = grid_bounding_box[1].z - grid_bounding_box[0].z;

    float x_voxel_width = x_width / GRID_X;
    float y_voxel_width = y_width / GRID_Y;
    float z_voxel_width = z_width / GRID_Z;

    float max_x = triangle[0].x > triangle[1].x ? triangle[0].x : triangle[1].x;
    max_x = max_x > triangle[2].x ? max_x : triangle[2].x;

    float max_y = triangle[0].y > triangle[1].y ? triangle[0].y : triangle[1].y;
    max_y = max_y > triangle[2].y ? max_y : triangle[2].y;

    float max_z = triangle[0].z > triangle[1].z ? triangle[0].z : triangle[1].z;
    max_z = max_z > triangle[2].z ? max_z : triangle[2].z;

    float min_x = triangle[0].x < triangle[1].x ? triangle[0].x : triangle[1].x;
    min_x = min_x < triangle[2].x ? min_x : triangle[2].x;

    float min_y = triangle[0].y < triangle[1].y ? triangle[0].y : triangle[1].y;
    min_y = min_y < triangle[2].y ? min_y : triangle[2].y;

    float min_z = triangle[0].z < triangle[1].z ? triangle[0].z : triangle[1].z;
    min_z = min_z > triangle[2].z ? min_z : triangle[2].z;

    x_index_st = floor(abs(grid_bounding_box[0].x - min_x) / x_voxel_width);
    x_index_ed = floor(abs(grid_bounding_box[0].x - max_x) / x_voxel_width);

    y_index_st = floor(abs(grid_bounding_box[0].y - min_y) / y_voxel_width);
    y_index_ed = floor(abs(grid_bounding_box[0].y - max_y) / y_voxel_width);

    z_index_st = floor(abs(grid_bounding_box[0].z - min_z) / z_voxel_width);
    z_index_ed = floor(abs(grid_bounding_box[0].z - max_z) / z_voxel_width);

    //clamp values
    x_index_st = min(x_index_ed, GRID_X - 1);
    y_index_st = min(y_index_ed, GRID_Y - 1);
    z_index_st = min(z_index_ed, GRID_Z - 1);

    x_index_ed = max(x_index_st, 0);
    y_index_ed = max(y_index_st, 0);
    z_index_ed = max(z_index_st, 0);
}

void Scene::createGrids()
{
    for (int i = 0; i < meshes.size(); i++)
    {
        Grid grid;
        grid.meshIndex = i;
        vector<vector<int>> voxels_buffer(GRID_X * GRID_Y * GRID_Z);
        int triangle_size = 0;

        float x_width = meshes[i].bounding_box[1].x - meshes[i].bounding_box[0].x;
        float y_width = meshes[i].bounding_box[1].y - meshes[i].bounding_box[0].y;
        float z_width = meshes[i].bounding_box[1].z - meshes[i].bounding_box[0].z;

        float x_voxel_width = x_width / GRID_X;
        float y_voxel_width = y_width / GRID_Y;
        float z_voxel_width = z_width / GRID_Z;

        for (int t = meshes[i].triangleDataIndices.start_index; t < meshes[i].triangleDataIndices.end_index; t++)
        {
            int t0_index = triangles[t].indices[0] + meshes[i].vertexDataIndices.start_index;
            int t1_index = triangles[t].indices[1] + meshes[i].vertexDataIndices.start_index;
            int t2_index = triangles[t].indices[2] + meshes[i].vertexDataIndices.start_index;

            glm::vec3 triangle[3];
            triangle[0] = vertexDataArr[t0_index].vertex;
            triangle[1] = vertexDataArr[t1_index].vertex;
            triangle[2] = vertexDataArr[t2_index].vertex;

            float max_x = triangle[0].x > triangle[1].x ? triangle[0].x : triangle[1].x;
            max_x = max_x > triangle[2].x ? max_x : triangle[2].x;

            float max_y = triangle[0].y > triangle[1].y ? triangle[0].y : triangle[1].y;
            max_y = max_y > triangle[2].y ? max_y : triangle[2].y;

            float max_z = triangle[0].z > triangle[1].z ? triangle[0].z : triangle[1].z;
            max_z = max_z > triangle[2].z ? max_z : triangle[2].z;

            float min_x = triangle[0].x < triangle[1].x ? triangle[0].x : triangle[1].x;
            min_x = min_x < triangle[2].x ? min_x : triangle[2].x;

            float min_y = triangle[0].y < triangle[1].y ? triangle[0].y : triangle[1].y;
            min_y = min_y < triangle[2].y ? min_y : triangle[2].y;

            float min_z = triangle[0].z < triangle[1].z ? triangle[0].z : triangle[1].z;
            min_z = min_z > triangle[2].z ? min_z : triangle[2].z;

            int x_index_st = floor(abs(meshes[i].bounding_box[0].x - min_x) / x_voxel_width);
            int x_index_ed = floor(abs(meshes[i].bounding_box[0].x - max_x) / x_voxel_width);

            int y_index_st = floor(abs(meshes[i].bounding_box[0].y - min_y) / y_voxel_width);
            int y_index_ed = floor(abs(meshes[i].bounding_box[0].y - max_y) / y_voxel_width);

            int z_index_st = floor(abs(meshes[i].bounding_box[0].z - min_z) / z_voxel_width);
            int z_index_ed = floor(abs(meshes[i].bounding_box[0].z - max_z) / z_voxel_width);

            //clamp values
            x_index_ed = min(x_index_ed, GRID_X - 1);
            y_index_ed = min(y_index_ed, GRID_Y - 1);
            z_index_ed = min(z_index_ed, GRID_Z - 1);

            x_index_st = max(x_index_st, 0);
            y_index_st = max(y_index_st, 0);
            z_index_st = max(z_index_st, 0);

            for (int z = z_index_st; z <= z_index_ed; z++)
            {
                for (int y = y_index_st; y <= y_index_ed; y++)
                {
                    for (int x = x_index_st; x <= x_index_ed; x++)
                    {
                        int index = x + y * GRID_X + GRID_X * GRID_Y * z;
                        voxels_buffer[index].push_back(t);
                        triangle_size++;
                    }
                }
            }
        }

        grid.voxelIndices.start_index = voxels.size();
        for (int i = 0; i < voxels_buffer.size(); i++)
        {
            Range range;
            range.start_index = perVoxelDataPool.size();
            for (int j = 0; j < voxels_buffer[i].size(); j++)
            {
                perVoxelDataPool.push_back(voxels_buffer[i][j]);
            }       
            range.end_index = perVoxelDataPool.size();
            voxels.push_back(range);
        }
        grid.voxelIndices.end_index = voxels.size();

        grids.push_back(grid);
    }
}