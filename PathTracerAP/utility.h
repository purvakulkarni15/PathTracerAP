#pragma once
#define GLM_FORCE_CUDA
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/random.h>
#include <thrust/remove.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define IS_EQUAL(x, y) (ABS((x) - (y)) < EPSILON)
#define IS_LESS_THAN(x, y) ((x) < (y) - EPSILON)
#define IS_MORE_THAN(x, y) ((x) > (y) + EPSILON)
#define CLAMP(value, min, max) ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))
#define CEIL(x,y) (((x) + (y) - 1) / (y))
#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f


void printCUDAMemoryInfo()
{
    size_t free_bytes, total_bytes;

    cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (cuda_status != cudaSuccess) 
    {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(cuda_status) << std::endl;
        return;
    }

    // Print the memory information
    std::cout << "Free GPU memory: " << free_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total GPU memory: " << total_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

}


__inline__ __host__ __device__
inline unsigned int utilHash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}



__inline__ __host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilHash((1 << 31) | (depth << 22) | iter) ^ utilHash(index);
    return thrust::default_random_engine(h);
}

__inline__ __host__ __device__ glm::vec3 reflectRay(const glm::vec3& incident_direction, const glm::vec3& normal_vector)
{

    glm::vec3 reflected_direction = normal_vector - 2.0f * glm::dot(incident_direction, normal_vector) * normal_vector;
    return reflected_direction;
}

__inline__ __host__ __device__ glm::vec3 transformDirection(glm::vec3& direction, const glm::mat4& matrix)
{
    return glm::vec3(matrix * glm::vec4(direction, 0.0f));
}


__inline__ __host__ __device__ glm::vec3 transformPosition(const glm::vec3& position, const glm::mat4& matrix)
{
    return glm::vec3(matrix * glm::vec4(position, 1.0f));
}

__inline__ __host__ __device__ glm::vec3 transformNormal(glm::vec3& normal, const glm::mat4& matrix)
{
    glm::mat3 upper_left_matrix = glm::mat3(matrix);
    glm::mat3 inverse_transpose = glm::transpose(glm::inverse(upper_left_matrix));

    return inverse_transpose * normal;
}


__forceinline__ __host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    glm::vec3 directionNotNormal;
    if (ABS(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (ABS(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
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

__forceinline__ __host__ __device__
glm::vec3 calculateCoatScattering(glm::vec3 normal, glm::vec3 ray_dir, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float rouletteRandomFloat = u01(rng);
    float threshold = 0.5f;
    glm::vec3 specularColor = glm::vec3(1, 1, 1);
    bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_Vec3f(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

    if (reflectFromSurface) 
    {
        return  reflectRay(ray_dir, normal);
    }

    else 
    {
        return calculateRandomDirectionInHemisphere(normal, rng);
    }
}

__forceinline__ __host__ __device__
glm::vec3 calculateMetalScattering(glm::vec3 normal, glm::vec3 ray_dir, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // compute random perturbation of ideal reflection vector
             // the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
    float phi = TWO_PI * u01(rng);
    float r2 = u01(rng);
    float phongexponent = 30;
    float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
    float sinTheta = sqrtf(1 - cosTheta * cosTheta);

    // create orthonormal basis uvw around reflection vector with hitpoint as origin 
    // w is ray direction for ideal reflection
    glm::vec3 w = glm::normalize(ray_dir - normal * 2.0f * glm::dot(normal, ray_dir));
    glm::vec3 u = glm::normalize(glm::cross((ABS(w.x) > .1 ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), w));
    glm::vec3 v = cross(w, u); // v is already normalised because w and u are normalised

    // compute cosine weighted random ray direction on hemisphere 
    return u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
}
