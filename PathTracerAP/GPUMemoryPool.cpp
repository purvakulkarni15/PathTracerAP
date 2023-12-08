#include "GPUMemoryPool.h"

template <typename T>
GPUMemoryPool<T>* GPUMemoryPool<T>::instance = nullptr;

template <typename T>
static GPUMemoryPool<T>* GPUMemoryPool<T>::getInstance()
{
	if (instance == nullptr)
	{
		instance = new GPUMemoryPool();
	}
	return instance;
}

template <typename T>
static void GPUMemoryPool<T>::allocate(const vector<T>& data)
{
	if (instance)
	{
		size = data.size();
		pool = (T*)malloc(sizeof(T) * size);

		const T* pData = data.data();
		std::copy(pData, pData + size, pool);
	}
}

template <typename T>
static void GPUMemoryPool<T>::free()
{
	delete pool;
	if (instance) delete instance;
}