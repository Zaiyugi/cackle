/* --- ExecutionPolicy ---
 * Execution policy used for CUDA kernel launching
 */

#ifndef __EXECUTIONPOLICY_CUH__
#define __EXECUTIONPOLICY_CUH__

#include <iostream>
#include <cstdlib>
#include <cuda.h>

namespace akasha
{

namespace util
{

class ExecutionPolicy
{
   public:
      ExecutionPolicy() : _block_size(), _grid_size() {}

      ExecutionPolicy(size_t b_x, size_t g_x) : _block_size(b_x), _grid_size(g_x) {}

      ExecutionPolicy(size_t b_x, size_t b_y, size_t g_x, size_t g_y) : 
         _block_size(b_x, b_y), 
         _grid_size(g_x, g_y)
      {}

      ExecutionPolicy(size_t b_x, size_t b_y, size_t b_z, size_t g_x, size_t g_y, size_t g_z) : 
         _block_size(b_x, b_y, b_z), 
         _grid_size(g_x, g_y, g_z)
      {}

      ExecutionPolicy(ExecutionPolicy& ep) : 
         _block_size(ep._block_size),
         _grid_size(ep._grid_size)
      {}

      const dim3& blockSize() const { return _block_size; }
      const dim3& gridSize() const { return _grid_size; }

   private:
      dim3 _block_size;
      dim3 _grid_size;
};

}

}

#endif
