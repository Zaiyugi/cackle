## Cackle: GPU Accelerated Path-tracer for 3D fractals
### Author: Zachary Shore
### Project: Digital Production Arts Thesis project

### Description

Cackle is a path-tracer for rendering 3D fractals using sphere-tracing. Rendering is done using a set of CUDA kernels. Results can be found here: zaiyugi.com/pages/cackle

### Building
*  make -- make all required libraries
*  make genswig -- makes the Python bindings; requires libraries to be built

### Directories
*  _hdr/_  
   Contains HDR maps 
*  _include/_  
   Contains class header files;
*  _lib/_  
   Contains static library files: libCackle.a
*  _src/_  
   Contains some source files
*  _obj/_  
   Intermediate object file storage
*  _python/_  
   Render and utility scripts
*  _swig/_  
   Swig interface and Swig-generated Python bindings

Created: 2017-04-01 
Edited: 2018-04-27
 
