/* --- DataObject class declaration ---
 * Spec: Class for managing both host and device pointers
 */

#ifndef __DATAOBJECT_H__
#define __DATAOBJECT_H__

#include <iostream>
#include <memory>
#include <cstring>
#include <utility>

// Forward declarations of DataObject
#include "DataObjectDefs.h"

#include "MemUtils.cuh"
#include "CudaDeleter.h"

namespace akasha
{

/* Datatype Policies */
template < typename T >
struct DataObjectType
{
   typedef T hostDataType;
   typedef T deviceDataType;
};

/* Ownership Policies */
template <typename T>
struct CreateUnique
{
   typedef typename DataObjectType<T>::hostDataType hostDataType;
   typedef typename DataObjectType<T>::deviceDataType deviceDataType;

   typedef typename std::unique_ptr<hostDataType[]> hostPointerType;
   typedef typename std::unique_ptr<deviceDataType, util::CudaDeleter> devicePointerType;
};

/* Class for storing both a host and device pointer */
template < typename T, typename PointerOwnershipPolicy >
class DataObject
{
   public:
      // Data-types for host and device pointers
      typedef typename DataObjectType<T>::hostDataType hostDataType;
      typedef typename DataObjectType<T>::deviceDataType deviceDataType;

      typedef typename PointerOwnershipPolicy::hostPointerType hostPointerType;
      typedef typename PointerOwnershipPolicy::devicePointerType devicePointerType;

      static const int HOSTDEVICE = 0;
      static const int HOSTONLY = 1;
      static const int DEVICEONLY = 2;

      // Constructors
      DataObject() {}

      DataObject(size_t size, int ms = HOSTDEVICE) :
         _size(size), _memory_space(ms)
      {
         _host_ptr = nullptr;
         _device_ptr = nullptr;

         if( _memory_space != DEVICEONLY )
            _host_ptr = hostPointerType( util::hostMalloc<hostDataType>(_size) );

         if( _memory_space != HOSTONLY )
            _device_ptr = devicePointerType( util::deviceMalloc<deviceDataType>(_size) );
      }

      DataObject(DataObject&& mo) noexcept :
         _size(mo._size), _memory_space(mo._memory_space),
         _host_ptr( std::move(mo._host_ptr) ),
         _device_ptr( std::move(mo._device_ptr) )
      {}

      ~DataObject() { /*std::cerr << "NOTE: Destroying DataObject" << std::endl;*/ }

      // Methods

      void mirror()
      {
         if( _memory_space == HOSTDEVICE )
         {
            std::cerr << "WARNING: DataObject already contains both host-side and device-side memory" << std::endl;
         } else if( _memory_space == DEVICEONLY )
         {
            std::cerr << "NOTE: Mirroring to host" << std::endl;
            _host_ptr = hostPointerType( util::hostMalloc<hostDataType>(_size) );
            _memory_space = HOSTDEVICE;
         } else if( _memory_space == HOSTONLY )
         {
            std::cerr << "NOTE: Mirroring to device" << std::endl;
            _device_ptr = devicePointerType( util::deviceMalloc<deviceDataType>(_size) );
            _memory_space = HOSTDEVICE;
         }
      }

      void resize(size_t new_size)
      {
         if( _memory_space == DEVICEONLY )
         {
            std::cerr << "ERROR: Cannot resize device only DataObject" << std::endl;
            return;
         }

         // Get current pointer
         hostDataType* curr = _host_ptr.get();

         // Allocate larger size, then copy over old contents
         hostDataType* new_ptr = util::hostMalloc<hostDataType>(new_size);
         std::memcpy(new_ptr, curr, _size * sizeof(hostDataType));

         // Delete old content, then give _host_ptr ownership of new_ptr
         _host_ptr.reset(nullptr);
         _host_ptr = hostPointerType(new_ptr);
         _size = new_size;

         if( _memory_space != HOSTONLY )
         {
            _device_ptr.reset(nullptr);
            _device_ptr = devicePointerType( util::deviceMalloc<deviceDataType>(_size) );
         }
      }

      void extend(size_t amt)
      { resize(_size + amt); }

      void updateDevice()
      {
         if( _memory_space == HOSTONLY )
         {
            std::cerr <<
               "ERROR: "
               "DataObject is host only; "
               "use mirror() to allocate device memory"
               << std::endl;
            return;
         }

         util::memcpyHostToDevice( _device_ptr.get(), _host_ptr.get(), _size );
      }

      void updateHost()
      {
         if( _memory_space == DEVICEONLY )
         {
            std::cerr <<
               "ERROR: "
               "DataObject is device only; "
               "use mirror() to allocate host memory"
               << std::endl;
            return;
         }

         util::memcpyDeviceToHost( _host_ptr.get(), _device_ptr.get(), _size );
      }

      deviceDataType* copyHostToDevice()
      {
         if( _memory_space == HOSTONLY )
         {
            std::cerr <<
               "ERROR: "
               "DataObject is host only; "
               "use mirror() to allocate device memory"
               << std::endl;
            return nullptr;
         }

         return util::memcpyHostToDevice(
            _device_ptr.get(),
            _host_ptr.get(),
            _size
         );
      }

      hostDataType* copyDeviceToHost()
      {
         if( _memory_space == DEVICEONLY )
         {
            std::cerr <<
               "ERROR: "
               "DataObject is device only; "
               "use mirror() to allocate host memory"
               << std::endl;
            return nullptr;
         }

         return util::memcpyDeviceToHost(
            _host_ptr.get(),
            _device_ptr.get(),
            _size
         );
      }

      // Getters
      hostDataType* getHostPointer() const
      { return _host_ptr.get(); }

      deviceDataType* getDevicePointer() const
      { return _device_ptr.get(); }

      // Operators
      DataObject& operator=(DataObject&& mo)
      {
         _size = mo._size;
         _host_ptr = std::move(mo._host_ptr);
         _device_ptr = std::move(mo._device_ptr);
         return *this;
      }

      hostDataType* operator*() const
      { return _host_ptr.get(); }

      // Members
      size_t _size;
      int _memory_space;
      hostPointerType _host_ptr;

   private:
      devicePointerType _device_ptr;

};

/* Functors */
struct getHostPointer_functor
{
   template <typename T>
   void operator() (DataObject<T>& obj, T** nt)
   {
      *nt = obj.getHostPointer();
   }

   template <typename T>
   void operator() (T& obj, T* nt)
   {
      *nt = obj;
   }

};

struct getDevicePointer_functor
{
   template <typename T>
   void operator() (DataObject<T>& obj, T** nt)
   {
      *nt = obj.getDevicePointer();
   }

   template <typename T>
   void operator() (T& obj, T* nt)
   {
      *nt = obj;
   }

};

struct memcpyHostToDevice_functor
{
   template <typename T>
   void operator() (DataObject<T>& obj)
   {
      obj.updateDevice();
   }

   template <typename T>
   void operator() (T& obj)
   {
      //std::cout << obj << std::endl;
   }

};

struct memcpyDeviceToHost_functor
{
   template <typename T>
   void operator() (DataObject<T>& obj)
   {
      obj.updateHost();
   }

   template <typename T>
   void operator() (T& obj)
   {
      //std::cout << obj << std::endl;
   }

};

}

#endif
