#include <float.h>
#include <helper_math.h>
#include <math_constants.h>
#include "RenderData.h"

#include "cuda/math_utils.cu"
#include "cuda/fractals.cu"

using namespace akasha;

// Juliabox

__device__ float sdf(float4 p, float4& C, Uniforms& params)
{
 	return juliabox(p, C, params);
}

__device__ float sdf(float4 p, Uniforms& params)
{
	float4 C = make_float4(0.0f);
 	return juliabox(p, C, params);
}

// Mandelbox
/*
__device__ float sdf(float4 p, float4& C, Uniforms& params)
{
	return mandelbox(p * params.scalefactor, C, params) * params.inverse_scalefactor;
}

__device__ float sdf(float4 p, Uniforms& params)
{
	float4 C = make_float4(0.0f);
	return mandelbox(p * params.scalefactor, C, params) * params.inverse_scalefactor;
}
*/

__device__ float4 getNormal(float4 p, Uniforms& params)
{
	float4 h = make_float4(0.0f);
	float4 n;

	float dt = 0.2e-5;// 0.00002;
	h.x = dt; n.x = sdf(p + h, params) - sdf(p - h, params); h.x = 0.0f;
	h.y = dt; n.y = sdf(p + h, params) - sdf(p - h, params); h.y = 0.0f;
	h.z = dt; n.z = sdf(p + h, params) - sdf(p - h, params);
	n.w = 0.0f;

	float len = sqrtf(n.x*n.x+n.y*n.y+n.z*n.z);
	if(len < 0.000005)
		return unitize(n * 1.0/dt);
	return n/len;
}

__device__ float4 colorize(float t, float4 cm_a, float4 cm_b, float4 cm_c, float4 cm_d)
{
	float4 C;
	C.x = cm_a.x + cm_b.x * cos(2.0 * CUDART_PI_F * (cm_c.x * t + cm_d.x) );
	C.y = cm_a.y + cm_b.y * cos(2.0 * CUDART_PI_F * (cm_c.y * t + cm_d.y) );
	C.z = cm_a.z + cm_b.z * cos(2.0 * CUDART_PI_F * (cm_c.z * t + cm_d.z) );
	C.w = 0.0f;
	return C;
}

__device__ float4 ortho(float4 v)
{
    //  See : http://lolengine.net/blog/2013/09/21/picking-orthogonal-vector-combing-coconuts
    return (fabs(v.x) > fabs(v.z)) ? make_float4(-v.y, v.x, 0.0, 0.0)  : make_float4(0.0, -v.z, v.y, 0.0);
}

__device__ float fresnelSchlick(float4 V, float4 L, float F0)
{
	float4 H = unitize(L + V);
	float dotVH = fmaxf(0.0, dot3(V, H));
	return (F0 + (1. - F0) * powf(2., (-5.55473 * dotVH - 6.98316) * dotVH));
}

// Unoptimized Cook-Torrance with GGX BRDF (Source: Epic Games UE4)
__device__ float ggx(float4 N, float4 V, float4 L, float R, float F0)
{
	float alpha = R*R;
	float4 H = unitize(L + V);

	float dotNL = fmaxf(0.0, dot3(N, L));
	float dotNV = fmaxf(0.0, dot3(N, V));
	float dotNH = fmaxf(0.0, dot3(N, H));
	float dotVH = fmaxf(0.0, dot3(V, H));

	float a2 = alpha*alpha;
	float denom = dotNH * dotNH * (a2 - 1.) + 1.;
	float D = a2 / (CUDART_PI_F * denom * denom);

	float k = powf(R + 1., 2.) * 0.25;
	float gnl = dotNL * (1. - k) + k;
	float gnv = dotNV * (1. - k) + k;

	float G = dotNL * dotNV / (gnl * gnv);

	float F = F0 + (1. - F0) * powf(2., (-5.55473 * dotVH - 6.98316) * dotVH);

	return D * F * G / (4. * dotNL * dotNV);
}

__device__ float4 getSampleBiased(float4 dir, float power, curandState_t &state)
{
	dir = unitize(dir);
	float4 o1 = unitize(ortho(dir));
	float4 o2 = unitize(cross(dir, o1));
	float r1 = curand_uniform(&state);
	float r2 = curand_uniform(&state);
	r1 = r1 * 2.0f * CUDART_PI_F;
	r2 = powf(r2, 1.0f / (power + 1.0f));
	float oneminus = sqrtf(1.0f - r2 * r2);
	return unitize(cosf(r1) * oneminus * o1 + sinf(r1) * oneminus * o2 + r2 * dir);
}

__device__ float4 getConeSample(float4 dir, double extent, curandState_t &state)
{
	dir = unitize(dir);
	float4 o1 = unitize(ortho(dir));
	float4 o2 = unitize(cross(dir, o1));
	double r1 = curand_uniform(&state);
	double r2 = curand_uniform(&state);
	r1 = r1 * 2.0 * CUDART_PI_F;
	r2 = 1.0 - r2 * extent;
	double oneminus = sqrt(1.0 - r2 * r2);
	return unitize(cos(r1) * oneminus * o1 + sin(r1) * oneminus * o2 + r2 * dir);
}

__device__ float4 getSample(float4 dir, curandState_t &state)
{
	return getSampleBiased(dir, 0.0, state); // <- unbiased!
}

__device__ float4 getCosineWeightedSample(float4 dir, curandState_t &state)
{
	return getSampleBiased(dir, 1.0, state);
}

__device__ float4 getBackground( float4 rd, Uniforms& params, float4 *hdr_map )
{
	// return make_float4(0.0);

	float theta = CUDART_PI_F - acos(rd.y);
	float phi = atan2(rd.x, rd.z) + CUDART_PI_F + params.hdrmap_rotation * 2.0 * CUDART_PI_F;

	int w = params.hdrmap_width;
	int h = params.hdrmap_height;
	int px = w * (phi / (2.0 * CUDART_PI_F));
	int py = h * (theta / CUDART_PI_F);

	px = (px < 0) ? px + w : ((px >= w) ? px - w : px);
	py = (py < 0) ? py + h : ((py >= h) ? py - h : py);

	return hdr_map[px + py * w];

}

__device__ float4 getBaseColor(float4 orbit, Uniforms& params)
{
	float4 color;
	orbit = abs(orbit);
	double r = fmaxf(orbit.x, fmaxf(orbit.y, orbit.z));
	color = colorize(r, params.colormap_a, params.colormap_b, params.colormap_c, params.colormap_d);

	color.w = 1.0;
	return color;
}

__device__ float4 getMetalRough(float4 orbit, Uniforms& params)
{
	float4 color;
	double r = sqrtf(dot3(orbit, orbit)) / orbit.w;
	color = colorize(r, params.colormap_a, params.colormap_b, params.colormap_c, params.colormap_d);

	color.w = 1.0;
	return color;
}

__device__ float4 shade(float4 p, float4 rd, float4 orbit, Camera& camera, Uniforms& params)
{
	float4 lgt_eye = camera.eye;

	float4 color;
	// double s = orbit.x*orbit.x + orbit.y*orbit.y + orbit.z*orbit.z;
	// double r = s / sqrtf(s);

	// double r = sqrt(orbit.x*orbit.x + orbit.y*orbit.y + orbit.z*orbit.z) / orbit.w;

	orbit = abs(orbit);
	double r = fmaxf(orbit.x, fmaxf(orbit.y, orbit.z));
	// r = powf(r, orbit.w);
	color = colorize(r, params.colormap_a, params.colormap_b, params.colormap_c, params.colormap_d);

	float4 N = getNormal(p, params);
	float4 toLgt = unitize(lgt_eye - p);
	color = color * dot(N, toLgt);
	float4 H = unitize(toLgt + -rd);
	color += make_float4( pow(dot(H, N), params.specular_exponent) );
	color.w = 0.0;

	return color;
}

__device__ bool trace(float4 r0, float4 rd, double tmax, Uniforms& params)
{
	float4 p;
	double t = 0.0, d = 0.0;

	for(int i = 0; i < params.raymarch_iterations && t < tmax; ++i)
	{
		p = r0 + rd * t;

		d = sdf(p, params);
		d *= params.fudgefactor;

		if(i == 0) d *= (params.dither * random(rd.x, rd.y)) + (1.0 - params.dither);
		t += d;

        double epsModified = t * params.epsilon;
        if( d < epsModified )
        {
            return true;
        }
	}

	return false;
}

__device__ bool trace(float4 r0, float4 rd, double tmax, float4& hit, float4& hitNormal, float4& orbit, int& i, Uniforms& params)
{
	float4 p;
	double t = 0.0, d = 0.0;

	for(i = 0; i < params.raymarch_iterations && t < tmax; ++i)
	{
		p = r0 + rd * t;

		d = sdf(p, orbit, params);
		d *= params.fudgefactor;

		if(i == 0) d *= (params.dither * random(rd.x, rd.y)) + (1.0 - params.dither);
		t += d;

        double epsModified = t * params.epsilon;
        if( d < epsModified )
        {
            t -= (epsModified - d);
            
            hit = r0 + rd * t; hit.w = 0.0f;
            hitNormal = getNormal(hit, params);
            return true;
        }
	}

	hit = p;
	hitNormal = unitize(ortho(rd));

	return false;
}

__device__ float4 raymarch(float4 r0, float4 rd, curandState_t &state, Camera& camera, Uniforms& params, float4 *hdr_map, HDRPointLight* hdr_lights)
{
	float4 luminance = make_float4(0.0, 0.0, 0.0, 0.0);
	if( !params.use_path_tracing )
	{
		float4 hit, hit_normal, orbit_trap = make_float4(0.0, 0.0, 0.0, 1.0);
		int steps = 0;

		bool hit_success = trace(r0, rd, 1.e7, hit, hit_normal, orbit_trap, steps, params);
		if(hit_success)
			luminance = shade(hit, rd, orbit_trap, camera, params);
		luminance = max(luminance, 0.0f);

		// Fog
		{
			luminance = max(luminance, 0.0f);

			float4 depth = camera.eye - hit;
			double t = sqrt(dot3(depth, depth));
			if( !hit_success )
				t = 1.0e7;

			double fogStrength = 1.0 - exp(double(-t * params.fog_level));
			luminance = lerp(luminance, params.fog_color, fogStrength);
		}

		// Edge Glow
		{
			luminance = max(luminance, 0.0f);
			double g = double(steps) / double(params.raymarch_iterations);
			g = 1.0 - g;
			g = pow(g, params.glow_strength);
			luminance = lerp( make_float4(0.99f), luminance, g );
		}

		luminance.w = 1.0;
		if(!hit_success)
			luminance.w = 0.0;
	} else 
	{
		int steps = 0;
		float4 hit = make_float4(0.0), hit_normal = make_float4(0.0), orbit_trap;
		float4 atten = make_float4(1.0, 1.0, 1.0, 0.0);
		luminance = make_float4(0.0, 0.0, 0.0, 0.0);

		float albedo = params.albedo;
		int i = 0;
		for(i = 0; i < params.max_path_length; ++i)
		{
			if(trace(r0, rd, 1.e9, hit, hit_normal, orbit_trap, steps, params))
			{
				float r1 = curand_uniform(&state);
				if( r1 > params.reflectivity )
				{
					break;
				} else
				{
					/*
					Sometimes the hit surface can have a normal that faces
					away from the view vector. This breaks shading,
					so catch it and end the path.

					I don't know why this happens. -_-
					*/ 
					if(dot3(-rd, hit_normal) < 0.0)
						break;

					float4 view = -rd;

					// Cosine Weighted Importance Sampling
					rd = getCosineWeightedSample(hit_normal, state);
					r0 = hit + hit_normal * params.epsilon * 2.0;

					// R: roughness, G: metallic, B: nada
					float4 metal_rough = getBaseColor(orbit_trap, params);
					float4 material = getMetalRough(orbit_trap, params);
					float metalness = 1.0;//(metal_rough.x + metal_rough.y + metal_rough.z) * 0.3;
					atten *= lerp(make_float4(1.0), material, metalness) * albedo;

					// Accumulate direct lighting
					for(int k = 0; k < params.hdrlight_count; ++k)
					{
						float4 dir = hdr_lights[k].p - r0;
						double max_dist = sqrt(dot3(dir, dir));
						float4 sample_dir = unitize(dir);

						double cosinefactor = dot3(hit_normal, sample_dir);
						if( cosinefactor > 0.0 && !trace(r0, sample_dir, max_dist, params) )
						{
							float4 Li = hdr_lights[k].color * (1.0 / pow(max_dist, hdr_lights[k].falloff));

							// F0 of 0.04 is about average for dielectrics
							float ks = fresnelSchlick(view, sample_dir, 0.04);
							float4 diffuse = atten / CUDART_PI_F;
							float4 specular = atten * ggx(hit_normal, view, sample_dir, 0.3, 0.04);
							float4 dielectric = (diffuse * (1.0 - ks) + specular);

							// Metallics
							float4 metallic = material * ggx(hit_normal, view, sample_dir, 0.5, 1.0);

							luminance += lerp(dielectric, metallic, metalness) * cosinefactor * Li;
						}
					}

					// Sample skydome
					float4 sample_dir = getCosineWeightedSample(hit_normal, state);
					if( !trace(r0, sample_dir, 1.e9, params) )
						luminance += atten * getBackground(sample_dir, params, hdr_map);
				}
			} else
			{
				luminance += atten * getBackground(rd, params, hdr_map);
				break;
			}
		}

		// Bring negative luminance up to 0
		luminance = max(luminance, 0.0f);

		luminance.w = 1.0;

		// If we didn't hit anything straight from the camera
		if( i == 0 )
			luminance = make_float4(0.0, 0.0, 0.0, 0.0);
	}

	return luminance;
}

__global__ void kernel_render(
	float4 *pixels, 
	int tile_idx, int tile_idy,
	int tile_width, int tile_height,
	int img_width, int img_height,
	Camera camera, Uniforms params,
	float4 *hdr_map, HDRPointLight *hdr_lights,
	curandState_t *states
	)
{
	int samples = params.samples;
	uint gbl_tid = blockDim.x * blockIdx.x + threadIdx.x;

	float4 color = make_float4(0.0);

	// params.scalefactor = 1.0;
	// params.inverse_scalefactor = 1.0;

	if(gbl_tid < tile_width * tile_height)
	{
		// int tid_x = tid % tile_width;
		// int tid_y = tid / tile_width;

		// int x_in_img = tid_x + tile_idx * tile_width;
		// int y_in_img = tid_y + tile_idy * tile_height;

		unsigned long long ndx_in_img = (unsigned long long)(tile_idx) * (unsigned long long)(tile_width) + (unsigned long long)(gbl_tid);
		int x_in_img = ndx_in_img % (unsigned long long)(img_width);
		int y_in_img = ndx_in_img / (unsigned long long)(img_width);

		// Flip for tiff
		y_in_img = img_height - y_in_img - 1;

		// unsigned long long ndx_in_img = (unsigned long long)(y_in_img) * (unsigned long long)(img_width) + (unsigned long long)(x_in_img);
		unsigned long long img_size = (unsigned long long)(img_width) * (unsigned long long)(img_height);
		if( ndx_in_img < img_size )
		{
			float2 texcoord;
			texcoord.x = x_in_img;
			texcoord.y = y_in_img;

			float2 p_uv = texcoord / make_float2(img_width, img_height);
			p_uv = p_uv * 2.0f - 1.0f;

			float aspect_ratio = float(img_width) / float(img_height);
			float focal_length = 1.0f / tan(camera.fov / 2.0f * CUDART_PI_F / 180.0f);

			float4 rd, r0;
			float2 aa_uv = p_uv;

			// rd = params.epsilon * camera.view + camera.eye;

			float subpixel_u = (2.0 / img_width) / float(samples);
			float subpixel_v = (2.0 / img_height) / float(samples);

			for(int i = 0; i < samples; ++i)
			{
				for(int j = 0; j < samples; ++j)
				{
					aa_uv = p_uv + make_float2(i * subpixel_u, j * subpixel_v);
					rd = unitize(
						camera.view * focal_length +
						camera.up * aa_uv.y +
						camera.right * aa_uv.x * aspect_ratio
						);
					rd.w = 0.0;
					r0 = camera.eye; r0.w = 0.0;
					color += raymarch(r0, rd, states[gbl_tid], camera, params, hdr_map, hdr_lights);
				}
			}

		}
		pixels[gbl_tid] = color / float(samples*samples);
	}

}
