/* --- CudaDeleter ---
 * CUDA Deleter custom delete struct
 */

#ifndef __CUDADELETER_H__
#define __CUDADELETER_H__

namespace akasha
{

namespace util
{

struct CudaDeleter
{
   void operator()(void *p);
};

}

}

#endif

