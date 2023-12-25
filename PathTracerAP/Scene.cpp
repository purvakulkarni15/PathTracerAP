#include "Scene.h"
#define CLAMP(value, min, max) ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))
Scene::Scene(string config)
{

    Mesh bunny_mesh;
    addMesh("Input data\\stanford_bunny.obj", bunny_mesh);
    meshes.push_back(bunny_mesh);

    Mesh armadillo_mesh;
    addMesh("Input data\\stanford_armadillo.obj", armadillo_mesh);
    meshes.push_back(armadillo_mesh);

    Mesh box;
    addMesh("Input data\\enclosing_box.obj", box);
    meshes.push_back(box);

    Mesh dragon_mesh;
    addMesh("Input data\\stanford_dinosaur.obj", dragon_mesh);
    meshes.push_back(dragon_mesh);

    Mesh ceiling_light;
    addMesh("Input data\\ceiling_light.obj", ceiling_light);
    meshes.push_back(ceiling_light);

    glm::mat4 scale_matrix, rotate_matrix, translation_matrix;

    Model bunny_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.08f, 0.08f, 0.08f));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(25.0f, -120.0f, 100.0f));
    bunny_model.mesh_index = 0;
    bunny_model.model_to_world = translation_matrix *rotate_matrix* scale_matrix;
    bunny_model.world_to_model = glm::inverse(bunny_model.model_to_world);
    bunny_model.mat.color = glm::vec3(0.001f, 0.99f, 0.2f);
    bunny_model.mat.material_type = Material::MaterialType::DIFFUSE;
    models.push_back(bunny_model);

    Model bunny_model_2;
    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.08f, 0.08f, 0.08f));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(-75.0f, -120.0f, 0.0f));
    bunny_model_2.mesh_index = 0;
    bunny_model_2.model_to_world = translation_matrix *rotate_matrix* scale_matrix;
    bunny_model_2.world_to_model = glm::inverse(bunny_model_2.model_to_world);
    bunny_model_2.mat.color = glm::vec3(0.99f, 0.99f, 0.001f);
    bunny_model_2.mat.material_type = Material::MaterialType::DIFFUSE;
    models.push_back(bunny_model_2);

    Model armadillo_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f, 0.1f, 0.1f));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(225.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(-250.0f, 5.0f, 0.0f));
    armadillo_model.mesh_index = 1;
    armadillo_model.model_to_world = translation_matrix *rotate_matrix* scale_matrix;
    armadillo_model.world_to_model = glm::inverse(armadillo_model.model_to_world);
    armadillo_model.mat.color = glm::vec3(0.001f, 0.001f, 0.99f);
    armadillo_model.mat.material_type = Material::MaterialType::DIFFUSE;
    models.push_back(armadillo_model);

    Model box_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.1, 0.1));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -120.0f, 0.0f));
    box_model.mesh_index = 2;
    box_model.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    box_model.world_to_model = glm::inverse(box_model.model_to_world);
    box_model.mat.color = glm::vec3(0.99f, 0.99f, 0.99f);
    box_model.mat.material_type = Material::MaterialType::DIFFUSE;
    models.push_back(box_model);

    Model dragon_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.1, 0.1));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(225.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(200.0f, 25.0f, 0.0f));
    dragon_model.mesh_index = 3;
    dragon_model.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    dragon_model.world_to_model = glm::inverse(dragon_model.model_to_world);
    dragon_model.mat.color = glm::vec3(0.001f, 0.5f, 0.99f);
    dragon_model.mat.material_type = Material::MaterialType::EMISSIVE;
    //models.push_back(dragon_model);

    Model light_model;

    scale_matrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.1, 0.1, 0.1));
    rotate_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    translation_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 750.0f, 0.0f));
    light_model.mesh_index = 4;
    light_model.model_to_world = translation_matrix * rotate_matrix * scale_matrix;
    light_model.world_to_model = glm::inverse(light_model.model_to_world);
    light_model.mat.color = glm::vec3(0.99f, 0.99f, 0.99f);
    light_model.mat.material_type = Material::MaterialType::EMISSIVE;
    models.push_back(light_model);

    generateUniformGrids();
}

void Scene::addMesh(string path, Mesh& mesh)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        cerr << "Error loading mesh: " << importer.GetErrorString() << endl;
        return;
    }

    processNode(scene->mRootNode, mesh, scene);

}

void Scene::processNode(aiNode* node, Mesh& mesh, const aiScene* scene)
{
    for (unsigned int i = 0; i < node->mNumMeshes; ++i)
    {
        aiMesh* ai_mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(ai_mesh, mesh, scene);
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i)
    {
        processNode(node->mChildren[i], mesh, scene);
    }
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
        //vertex_data_obj.normal = getFromVector3D(ai_mesh->mNormals[i]);
        vertex_data_pool.push_back(vertex_data_obj);
        mesh.bounding_box.update(vertex_data_obj.vertex);
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

void getVoxelIndex( Voxel3D& min, Voxel3D& max, 
                    Bounding_Box& bounding_box,
                    const float& x_voxel_width, const float& y_voxel_width, const float& z_voxel_width,
                    const glm::vec3* triangle)
{
    Bounding_Box t_box;
    t_box.update(triangle[0]);
    t_box.update(triangle[1]);
    t_box.update(triangle[2]);

    min.x = floor(abs(bounding_box.min.x - t_box.min.x) / x_voxel_width);
    max.x = floor(abs(bounding_box.min.x - t_box.max.x) / x_voxel_width);

    min.y = floor(abs(bounding_box.min.y - t_box.min.y) / y_voxel_width);
    max.y = floor(abs(bounding_box.min.y - t_box.max.y) / y_voxel_width);

    min.z = floor(abs(bounding_box.min.z - t_box.min.z) / z_voxel_width);
    max.z = floor(abs(bounding_box.min.z - t_box.max.z) / z_voxel_width);

    min.x = CLAMP(min.x, 0, GRID_X - 1);
    min.y = CLAMP(min.y, 0, GRID_Y - 1);
    min.z = CLAMP(min.z, 0, GRID_Z - 1);

    max.x = CLAMP(max.x, 0, GRID_X - 1);
    max.y = CLAMP(max.y, 0, GRID_Y - 1);
    max.z = CLAMP(max.z, 0, GRID_Z - 1);
}

void Scene::generateUniformGrids()
{
    vector<bool> is_mesh_processed(meshes.size(), false);
    vector<int> grid_index_cache(meshes.size());

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

        float x_width = meshes[grid.mesh_index].bounding_box.max.x - meshes[grid.mesh_index].bounding_box.min.x;
        float y_width = meshes[grid.mesh_index].bounding_box.max.y - meshes[grid.mesh_index].bounding_box.min.y;
        float z_width = meshes[grid.mesh_index].bounding_box.max.z - meshes[grid.mesh_index].bounding_box.min.z;

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

            Voxel3D min, max;

            getVoxelIndex(  min, max,
                            meshes[grid.mesh_index].bounding_box,
                            x_voxel_width, y_voxel_width, z_voxel_width,
                            triangle);

            for (int z = min.z; z <= max.z; z++)
            {
                for (int y = min.y; y <= max.y; y++)
                {
                    for (int x = min.x; x <= max.x; x++)
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
                TriangleIndex iTriangle;
                iTriangle.index = voxels_buffer[i][j];
                per_voxel_data_pool.push_back(iTriangle);
            }       
            range.end_index = per_voxel_data_pool.size();
            voxels.push_back(range);
        }
        grid.voxelIndices.end_index = voxels.size();

        grids.push_back(grid);
    }
}