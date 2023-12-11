#include "Scene.h"

Scene::Scene(string config)
{

    Mesh bunny_mesh;
    addMesh("Input data\\bunny.obj", bunny_mesh);
    meshes.push_back(bunny_mesh);

    Mesh armadillo_mesh;
    addMesh("Input data\\armadillo.obj", armadillo_mesh);
    meshes.push_back(armadillo_mesh);

    glm::mat4 scale_matrix, rotate_matrix, translation_matrix;

    Model bunny_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.05, 0.05, 0.05));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(135.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(-100.0, 0, 0));
    bunny_model.mesh_index = 0;
    bunny_model.model_to_world = rotate_matrix * scale_matrix;
    bunny_model.world_to_model = glm::inverse(bunny_model.model_to_world);
    bunny_model.mat.color = glm::vec3(0.0f, 250.0f, 0.0f);
    models.push_back(bunny_model);

    Model armadillo_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.1, 0.1));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(-100.0, 0, 0));
    armadillo_model.mesh_index = 1;
    armadillo_model.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    armadillo_model.world_to_model = glm::inverse(armadillo_model.model_to_world);
    armadillo_model.mat.color = glm::vec3(0.0f, 0.0f, 250.0f);
    models.push_back(armadillo_model);

    generateUniformGrids();
}

void Scene::addMesh(string path, Mesh& mesh)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices);

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
    vec3.x = vector3D.x * 1000.0;
    vec3.y = vector3D.y * 1000.0;
    vec3.z = vector3D.z * 1000.0;
    return vec3;
}

void Scene::processMesh(aiMesh* ai_mesh, Mesh& mesh, const aiScene* scene)
{
    mesh.vertexDataIndices.start_index = vertex_data_pool.size();
    for (unsigned int i = 0; i < ai_mesh->mNumVertices; ++i)
    {
        VertexData vertex_data_obj;
        vertex_data_obj.vertex = getFromVector3D(ai_mesh->mVertices[i]);
        vertex_data_obj.normal = getFromVector3D(ai_mesh->mNormals[i]);
        vertex_data_pool.push_back(vertex_data_obj);

        updateBoundingBox(mesh.bounding_box, vertex_data_obj.vertex);
    }
    mesh.vertexDataIndices.end_index = vertex_data_pool.size();

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

void getVoxelIndex( int& x_index_st, int& x_index_ed, 
                    int& y_index_st, int& y_index_ed, 
                    int& z_index_st, int& z_index_ed, 
                    const glm::vec3& bb_min, 
                    const float& x_width, const float& y_width, const float& z_width,
                    const float& x_voxel_width, const float& y_voxel_width, const float& z_voxel_width,
                    const glm::vec3 triangle[3])
{
    glm::vec3 t_box[2];

    t_box[1].x = triangle[0].x > triangle[1].x ? triangle[0].x : triangle[1].x;
    t_box[1].x = t_box[1].x > triangle[2].x ? t_box[1].x : triangle[2].x;

    t_box[1].y = triangle[0].y > triangle[1].y ? triangle[0].y : triangle[1].y;
    t_box[1].y = t_box[1].y > triangle[2].y ? t_box[1].y : triangle[2].y;

    t_box[1].z = triangle[0].z > triangle[1].z ? triangle[0].z : triangle[1].z;
    t_box[1].z = t_box[1].z > triangle[2].z ? t_box[1].z : triangle[2].z;

    t_box[0].x = triangle[0].x < triangle[1].x ? triangle[0].x : triangle[1].x;
    t_box[0].x = t_box[0].x < triangle[2].x ? t_box[0].x : triangle[2].x;

    t_box[0].y = triangle[0].y < triangle[1].y ? triangle[0].y : triangle[1].y;
    t_box[0].y = t_box[0].y < triangle[2].y ? t_box[0].y : triangle[2].y;

    t_box[0].z = triangle[0].z < triangle[1].z ? triangle[0].z : triangle[1].z;
    t_box[0].z = t_box[0].z > triangle[2].z ? t_box[0].z : triangle[2].z;

    x_index_st = floor(abs(bb_min.x - t_box[0].x) / x_voxel_width);
    x_index_ed = floor(abs(bb_min.x - t_box[1].x) / x_voxel_width);

    y_index_st = floor(abs(bb_min.y - t_box[0].y) / y_voxel_width);
    y_index_ed = floor(abs(bb_min.y - t_box[1].y) / y_voxel_width);

    z_index_st = floor(abs(bb_min.z - t_box[0].z) / z_voxel_width);
    z_index_ed = floor(abs(bb_min.z - t_box[1].z) / z_voxel_width);

    //clamp values
    x_index_ed = min(x_index_ed, GRID_X - 1);
    y_index_ed = min(y_index_ed, GRID_Y - 1);
    z_index_ed = min(z_index_ed, GRID_Z - 1);

    x_index_st = max(x_index_st, 0);
    y_index_st = max(y_index_st, 0);
    z_index_st = max(z_index_st, 0);
}

void Scene::generateUniformGrids()
{
    vector<bool> is_mesh_processed(meshes.size(), false);
    vector<bool> grid_index_cache(meshes.size());

    for (int i = 0; i < models.size(); i++)
    {
        if (is_mesh_processed[models[i].mesh_index])
        {
            models[i].grid_index = grid_index_cache[models[i].mesh_index];
            continue;
        }

        is_mesh_processed[models[i].mesh_index] = true;
        grid_index_cache[models[i].mesh_index] = grids.size();
        models[i].grid_index = grids.size();

        Grid grid;
        
        grid.mesh_index = models[i].mesh_index;
        
        vector<vector<int>> voxels_buffer(GRID_X * GRID_Y * GRID_Z);

        float x_width = meshes[grid.mesh_index].bounding_box[1].x - meshes[grid.mesh_index].bounding_box[0].x;
        float y_width = meshes[grid.mesh_index].bounding_box[1].y - meshes[grid.mesh_index].bounding_box[0].y;
        float z_width = meshes[grid.mesh_index].bounding_box[1].z - meshes[grid.mesh_index].bounding_box[0].z;

        float x_voxel_width = x_width / GRID_X;
        float y_voxel_width = y_width / GRID_Y;
        float z_voxel_width = z_width / GRID_Z;

        for (int t = meshes[grid.mesh_index].triangleDataIndices.start_index; t < meshes[grid.mesh_index].triangleDataIndices.end_index; t++)
        {
            int t0_index = triangles[t].indices[0];
            int t1_index = triangles[t].indices[1];
            int t2_index = triangles[t].indices[2];

            glm::vec3 triangle[3];
            triangle[0] = vertex_data_pool[t0_index].vertex;
            triangle[1] = vertex_data_pool[t1_index].vertex;
            triangle[2] = vertex_data_pool[t2_index].vertex;

            int x_index_st;
            int x_index_ed;
            int y_index_st;
            int y_index_ed;
            int z_index_st;
            int z_index_ed;

            getVoxelIndex(  x_index_st, x_index_ed, y_index_st, y_index_ed, z_index_st, z_index_ed, 
                            meshes[grid.mesh_index].bounding_box[0],
                            x_width, y_width, z_width,
                            x_voxel_width, y_voxel_width, z_voxel_width,
                            triangle);

            for (int z = z_index_st; z <= z_index_ed; z++)
            {
                for (int y = y_index_st; y <= y_index_ed; y++)
                {
                    for (int x = x_index_st; x <= x_index_ed; x++)
                    {
                        int index = x + y * GRID_X + GRID_X * GRID_Y * z;
                        voxels_buffer[index].push_back(t);
                    }
                }
            }
        }

        grid.voxelIndices.start_index = voxels.size();
        for (int i = 0; i < voxels_buffer.size(); i++)
        {
            Range range;
            range.start_index = per_voxel_data_pool.size();
            for (int j = 0; j < voxels_buffer[i].size(); j++)
            {
                per_voxel_data_pool.push_back(voxels_buffer[i][j]);
            }       
            range.end_index = per_voxel_data_pool.size();
            voxels.push_back(range);
        }
        grid.voxelIndices.end_index = voxels.size();

        grids.push_back(grid);
    }
}