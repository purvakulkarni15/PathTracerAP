#pragma once
#include <iostream>
#include <vector>

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
			instance = new GPUMemoryPool();
		}
		return instance;
	}
	void allocate(const vector<T>& data)
	{
		if (instance)
		{
			size = data.size();
			pool = (T*)malloc(sizeof(T) * size);

			const T* pData = data.data();
			std::copy(pData, pData + size, pool);
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