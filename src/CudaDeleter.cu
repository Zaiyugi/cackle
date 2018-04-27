/* --- CudaDeleter ---
 * CUDA Deleter custom memory delete struct
 */

#include "CudaDeleter.h"
#include <iostream>
#include <cuda.h>

namespace akasha
{

namespace util
{

void CudaDeleter::operator()(void *p)
{
     //std::cerr << "NOTE: CudaDeleter: Free..." << std::endl;
     cudaError_t res = cudaFree(p);
     if (res != cudaSuccess)
     {
         std::cerr << "ERROR: CUDA Error at delete: " << cudaGetErrorString(res) << std::endl;
     }
}

}

}
