#ifndef __RENDERDATA_H__
#define __RENDERDATA_H__

namespace akasha
{

#ifndef __CUDACC__

class float4
{
public:
   float4() : x(0.0), y(0.0), z(0.0), w(0.0) {}
   
   float4(float r, float g, float b, float a) : x(r), y(g), z(b), w(a) {}

   float4(float r, float g, float b) : x(r), y(g), z(b), w(0.0) {}

   float4(float4& flt) : x(flt.x), y(flt.y), z(flt.z), w(flt.w) {}

   float4& operator=(float4 flt)
   {
      x = flt.x; y = flt.y; z = flt.z; w = flt.w;
   }

   float x;
   float y;
   float z;
   float w;
};

#endif

class Camera
{
public:
   Camera()
   {}

   Camera(Camera& cam)
   {
      eye = cam.eye;
      aim = cam.aim;

      view = cam.view;
      up = cam.up;
      right = cam.right;
      
      fov = cam.fov;
   }

   float4 eye;
   float4 aim;

   float4 view;
   float4 up;
   float4 right;
   float fov;
};

class HDRPointLight
{
public:
   HDRPointLight()
   {}

   HDRPointLight(HDRPointLight& lgt)
   {
      p = lgt.p;
      color = lgt.color;
      falloff = lgt.falloff;
      px = lgt.px;
      py = lgt.py;
   }

   float4 p;
   float4 color;
   double falloff;
   int px;
   int py;
};

class Uniforms
{
public:
   Uniforms()
   {}

   Uniforms(Uniforms& unis)
   {
      use_path_tracing = unis.use_path_tracing;
      max_path_length = unis.max_path_length;
      samples = unis.samples;
      dither = unis.dither;
      fudgefactor = unis.fudgefactor;

      fractal_iterations = unis.fractal_iterations;
      raymarch_iterations = unis.raymarch_iterations;
      flimit = unis.flimit;
      epsilon = unis.epsilon;
      scale = unis.scale;
      mr2 = unis.mr2;
      fr2 = unis.fr2;

      julia_c = unis.julia_c;
      julia_offset = unis.julia_offset;
      de_offset = unis.de_offset;

      fog_level = unis.fog_level;
      fog_color = unis.fog_color;
      glow_strength = unis.glow_strength;
      
      u_time = unis.u_time;

      scalefactor = unis.scalefactor;
      inverse_scalefactor = unis.inverse_scalefactor;
      bv_radius = unis.bv_radius;

      specular_exponent = unis.specular_exponent;
      gamma = unis.gamma;

      reflectivity = unis.reflectivity;
      albedo = unis.albedo;

      colormap_a = unis.colormap_a;
      colormap_b = unis.colormap_b;
      colormap_c = unis.colormap_c;
      colormap_d = unis.colormap_d;

      hdrmap_width = unis.hdrmap_width;
      hdrmap_height = unis.hdrmap_height;
      hdrmap_rotation = unis.hdrmap_rotation;
      hdrlight_count = unis.hdrlight_count;
   }

   bool use_path_tracing;

   int max_path_length;
   int samples;
   float dither;
   float fudgefactor;
   
   int fractal_iterations;
   int raymarch_iterations;
   float4 flimit;
   float epsilon;
   float scale;
   float mr2;
   float fr2;

   float4 julia_c;
   float4 julia_offset;
   float de_offset;

   float fog_level;
   float4 fog_color;
   float glow_strength;
   
   float u_time;

   float scalefactor;
   float inverse_scalefactor;
   float bv_radius;
   float specular_exponent;
   float gamma;

   float reflectivity;
   float albedo;

   float4 colormap_a;
   float4 colormap_b;
   float4 colormap_c;
   float4 colormap_d;

   int hdrmap_width;
   int hdrmap_height;
   float hdrmap_rotation;
   int hdrlight_count;
};

}

#endif
