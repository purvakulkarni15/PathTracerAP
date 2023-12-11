#pragma once
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

using namespace std;

template <typename T>
class GPUMemoryPool
{
public:
	static GPUMemoryPool* instance;
	static GPUMemoryPool* getInstance()
	{
		if (instance == nullptr)
		{
			//instance = new GPUMemoryPool();
			cudaMallocManaged(&instance, sizeof(GPUMemoryPool<T>));
		}
		return instance;
	}
	void allocate(const vector<T>& data)
	{
		if (instance)
		{
			size = data.size();
			cudaError_t err = cudaMallocManaged(&pool, sizeof(T)*size);
			//pool = (T*)malloc(sizeof(T) * size);

			const T* pData = data.data();
			//std::copy(pData, pData + size, pool);
			err = cudaMemcpy(pool, pData, sizeof(T) * size, cudaMemcpyHostToDevice);
		}
	}
	void free()
	{
		delete pool;
		if (instance) delete instance;
	}

	int size;
	T* pool;

private:
	GPUMemoryPool() :size(0), pool(nullptr) {};
};

template <typename T>
GPUMemoryPool<T>* GPUMemoryPool<T>::instance = nullptr;