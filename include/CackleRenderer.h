#ifndef __OGLRENDERER_H__
#define __OGLRENDERER_H__

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <random>
#include <memory>
#include <limits>

#include <cmath>
#include <ctime>

#include "DataObjectDefs.h"
#include "RenderData.h"

// #include "Image.h"
// #include "OIIOFiles.h"

#include <OpenImageIO/imageio.h>
// #include <OpenEXR/half.h>

namespace akasha
{

class CackleRenderer
{
   
   public:

      CackleRenderer() :
         canvas_width(15), canvas_height(8.4375), dpi(128),
         tile(nullptr), image(nullptr), 
         camera(nullptr), uniforms(nullptr),
         hdr_map(nullptr), hdr_lights(nullptr)
      {}

      CackleRenderer(int w, int h, int p, int mtw = 32768) :
         canvas_width(w), canvas_height(h), dpi(p), max_tile_width(mtw),
         tile(nullptr), image(nullptr),
         camera(nullptr), uniforms(nullptr),
         hdr_map(nullptr), hdr_lights(nullptr)
      {}

      ~CackleRenderer();

      void init();
      void initCubemap();

      void render();
      void renderChunk(int cx, int n_chunks);
      void renderCubemap();

      void outputImage(const char* filepath);
      void outputCubemap(const char* filepath);

      void setCameraEye(float x, float y, float z);
      void setCameraAim(float x, float y, float z);
      void setCameraView(float x, float y, float z);
      void setCameraUp(float x, float y, float z);
      void setCameraRight(float x, float y, float z);
      void setCameraFOV(float fov);

      void usePathTracing(bool upt);

      void setMaxPathLength(int mpl);
      void setSampleCount(int sc);

      void setDither(float dither);
      void setFudgeFactor(float fudgefactor);

      void setFractalIterations(int fi);
      void setRaymarchIterations(int ri);

      void setFLimit(float x, float y, float z);
      void setJuliaC(float x, float y, float z);
      void setJuliaOffset(float x, float y, float z);
      void setDEOffset(float deo);

      void setEpsilon(float eps);
      void setScale(float scale);
      void setMR2(float mr2);
      void setFR2(float fr2);

      void setFogLevel(float fog);
      void setFogColor(float r, float g, float b);
      
      void setGlowStrength(float glow);
      void setSpecularExponent(float spec);
      void setGamma(float g);
      void setReflectivity(float refl);
      void setAlbedo(float a);

      void addPointLight(float x, float y, float z, float r, float g, float b, float falloff);

      void setColormapA(float r, float g, float b);
      void setColormapB(float r, float g, float b);
      void setColormapC(float r, float g, float b);
      void setColormapD(float r, float g, float b);
      void setColormap(int id);

      void setHDRMapRotation(float mr);
      void setHDRLightCount(int lc);
      void setHDRMap(const char* path);

   private:

      void printInfo();
      void renderImage(float* img);
      void renderTile(int tx, int ty);

      int canvas_width;
      int canvas_height;
      int dpi;

      int max_tile_width;
      int tile_width;
      int tile_height;

      std::string hdr_map_path;
      DataObject<float4>* hdr_map;      
      DataObject<HDRPointLight>* hdr_lights;
      int n_hdr_lights;

      DataObject<float4>* tile;
      std::unique_ptr<float[]> image; // Uses OpenEXR 16-bit floating point (half)
      int img_width;
      int img_height;
      unsigned long long ull_img_size;

      std::unique_ptr<Camera> camera;
      std::unique_ptr<Uniforms> uniforms;
};

}

#endif
