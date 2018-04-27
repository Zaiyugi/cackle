#ifndef __FRACTALS_CU__
#define __FRACTALS_CU__

#include <float.h>
#include <helper_math.h>
#include <math_constants.h>

#include "RenderData.h"
#include "math_utils.cu"

#define USE_INF_NORM

using namespace akasha;

__device__ void sphereFold(float4& z, float MR2, float FR2)
{
	float r2 = dot3(z, z);

	float t = 1.0;
	if( r2 < MR2 )
		t = FR2 / MR2;
	else if( r2 < FR2 )
		t = FR2 / r2;

	z *= t;
/*
	z *= lerp(
		1.0f / MR2, // Linear scaling inside
		lerp(
			1.0f / r2, // Sphere inversion outside
			1.0f,
			step(1.0f, r2)
		),
		step(MR2, r2)
	);
*/
}
 
__device__ void boxFold(float4& z, float4 flimit)
{
	z.x = clamp(z.x, -flimit.x, flimit.x) * 2.0 - z.x;
	z.y = clamp(z.y, -flimit.y, flimit.y) * 2.0 - z.y;
	z.z = clamp(z.z, -flimit.z, flimit.z) * 2.0 - z.z;
}

__device__ float Menger(float4 z)
{
    // float r;
    int n = 0;

    float4 MnOffset = make_float4(1.4898, 1.95918, 1.10202, 0.0);
    float MnScale = 3.0;

	z = fabs(z);
	if(z.x < z.y)
	{
		cuswap(z.x, z.y);
	}
	if(z.x < z.z)
	{
		cuswap(z.x, z.z);
	}
	if(z.y < z.z)
	{
		cuswap(z.y, z.z);
	}
	if(z.z < 1. / 3.)
    {
    	z.z -= 2. * (z.z - 1. / 3. );
    }
    
    while (n < 5 && dot3(z, z) < 100.0)
    {        
        z = MnScale * (z - MnOffset) + MnOffset;
        
        // Fold
        z = fabs(z);
        if(z.x < z.y)
       	{
       		cuswap(z.x, z.y);
       	}
        if(z.x < z.z)
        {
        	cuswap(z.x, z.z);
        }
        if(z.y < z.z)
        {
        	cuswap(z.y, z.z);
        }
        if(z.z < 1. / 3. * MnOffset.z)
        {
        	z.z -= 2. * (z.z - 1. / 3. * MnOffset.z);
        }
        
        n++;
    }
    
    return float(z.x-MnOffset.x) * pow(MnScale, float(-n));
}

__device__ float mandelbox(float4 z, float4& orbitTrap, Uniforms& params)
{
	float4 p = z; p.w = 1.0;
	float4 p0 = z; p0.w = 1.0;
	orbitTrap = make_float4(1e20f);

	float r2;

#ifdef USE_INF_NORM
	float4 p2 = fabs(p);
	r2 = fmaxf(p2.x, fmaxf(p2.y, p2.z));
#else
	r2 = dot3(p, p);
#endif

	for(int i = 0; i < params.fractal_iterations && r2 < 100.0; i++)
	{
		boxFold(p, params.flimit);
		sphereFold(p, params.mr2, params.fr2);

		p.x = p.x * params.scale + p0.x;
		p.y = p.y * params.scale + p0.y;
		p.z = p.z * params.scale + p0.z;
		p.w = p.w * fabs(params.scale) + p0.w;

#ifdef USE_INF_NORM
		float4 p2 = fabs(p);
		r2 = fmaxf(p2.x, fmaxf(p2.y, p2.z));
#else
		r2 = dot3(p, p);
#endif

		orbitTrap.x = fminf(fabs(p.x), orbitTrap.x);
		orbitTrap.y = fminf(fabs(p.y), orbitTrap.y);
		orbitTrap.z = fminf(fabs(p.z), orbitTrap.z);
		orbitTrap.w = fminf( r2, orbitTrap.w );
	}

	// return (sqrtf(dot3(p, p)) - params.bv_radius) / fabs(p.w);
	//return (sqrtf(dot3(p, p))) / fabs(p.w);

	float4 b = make_float4(2.0); b.w = 0.0;
	b = fmaxf(fabs(p) - b, make_float4(0.0f));
	// return (sqrtf(dot3(b, b)) - params.bv_radius) / fabs(p.w);
	return fabs( (sqrtf(dot3(b, b)) - params.bv_radius) / p.w );
}

__device__ float juliabox(float4 z, float4& orbitTrap, Uniforms& params)
{
	float4 p = z; p.w = 1.0;
	float4 C = params.julia_c; //make_float4(1.97016, -0.03052, -0.1194, 0.0);
	//float4 flimit = make_float4(1.0084, 1.0, 1.0084, 0.0);

	orbitTrap = make_float4(1e20f);
	float DEfactor = 1.0;
	float DEoffset = params.de_offset; //0.00262;
	float4 Offset = params.julia_offset; //make_float4(0.55552, 0.48148, -0.1852, 0.0);

	float r2;
	
	float4 p2 = fabs(p);
	r2 = fmaxf(p2.x, fmaxf(p2.y, p2.z));

	for(int i = 0; i < params.fractal_iterations && r2 < 60.0; i++)
	{
		p.x = clamp(p.x, -params.flimit.x, params.flimit.x) * 2.0 - p.x;
		p.y = clamp(p.y, -params.flimit.y, params.flimit.y) * 2.0 - p.y;
		p.z = clamp(p.z, -params.flimit.z, params.flimit.z) * 2.0 - p.z;
	
		float r2 = dot3(p, p);
		float k = fmaxf(params.scale/r2, 1.0);

		p.x = p.x * k + C.x;
		p.y = p.y * k + C.y;
		p.z = p.z * k + C.z;
		DEfactor *= k;

		float4 p2 = fabs(p);
		r2 = fmaxf(p2.x, fmaxf(p2.y, p2.z));

		orbitTrap.x = fminf(p2.x, orbitTrap.x);
		orbitTrap.y = fminf(p2.y, orbitTrap.y);
		orbitTrap.z = fminf(p2.z, orbitTrap.z);
		orbitTrap.w = fminf(r2, orbitTrap.w);
	}

	return fabs(0.5 * Menger(p - Offset) / DEfactor - DEoffset);	
}

#endif
