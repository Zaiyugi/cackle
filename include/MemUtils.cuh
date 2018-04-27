/* --- MemUtils ---
 * Memory utility functions
 */

#ifndef __MEMUTILS_CUH__
#define __MEMUTILS_CUH__

#include <iostream>
#include <cstdlib>
#include <cuda.h>

namespace akasha
{

namespace util
{

template <typename T>
T* hostMalloc(size_t size)
{
   return new T[size];
}

template <typename T>
T* deviceMalloc(size_t size)
{
   T* pMemory = nullptr;
   cudaError_t res = cudaMalloc(&pMemory, size * sizeof(T));
   if (res != cudaSuccess)
   {
      std::cout << "CUDA Error: " << cudaGetErrorString(res) << std::endl;
      // throw; // No exception handling for now
   }
   return pMemory;
}

template <typename T>
T* memcpyDeviceToHost(T* dest, T* src, size_t size)
{
   cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToHost);
   return dest;
}

template <typename T>
T* memcpyHostToDevice(T* dest, T* src, size_t size)
{
   cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyHostToDevice);
   return dest;
}

}

}

#endif
