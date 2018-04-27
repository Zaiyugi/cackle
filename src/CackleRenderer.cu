#include "CackleRenderer.h"

#include "DataObject.h"
#include "ExecutionPolicy.cuh"
#include "ProgressMeter.h"

#include "cuda/raymarchKernel.cu"
#include "cuda/math_utils.cu"

OIIO_NAMESPACE_USING

namespace akasha
{

CackleRenderer::~CackleRenderer()
{
	if(tile)
		delete tile;

	if(hdr_map)
		delete hdr_map;

	if(hdr_lights)
		delete hdr_lights;

	std::cout << "LOG: CackleRenderer deleted..." << std::endl;
	cudaDeviceReset();
}

void CackleRenderer::init()
{
	camera = std::unique_ptr<Camera>(new Camera());
	uniforms = std::unique_ptr<Uniforms>(new Uniforms());

	// Round dpi to next multiple of 128
	if(dpi % 128)
		dpi = dpi + (128 - dpi % 128);

	// Round canvas_width and canvas_height to nearest even value
	if(canvas_width % 2)
		canvas_width = canvas_width + (2 - canvas_width % 2);

	if(canvas_height % 2)
		canvas_height = canvas_height + (2 - canvas_height % 2);

	img_width = std::floor(canvas_width * dpi);
	img_height = std::floor(canvas_height * dpi);

	ull_img_size = static_cast<unsigned long long>(img_width) * static_cast<unsigned long long>(img_height) * 4ULL;
	image = std::unique_ptr<float[]>(new float[ull_img_size]);

	for(unsigned long long i = 0; i < ull_img_size / 4ULL; i++)
	{
		image[i] = 0.0;
		image[i+1ULL] = 0.0;
		image[i+2ULL] = 0.0;
		image[i+3ULL] = 1.0;
	}

	if(img_width * img_height < max_tile_width)
		tile_width = img_width * img_height;
	else
		tile_width = max_tile_width;

	tile_height = 1;
	tile = new DataObject<float4>(tile_width * tile_height);

	// Init HDR map
	n_hdr_lights = 1;
	uniforms->hdrlight_count = n_hdr_lights;
	uniforms->hdrmap_rotation = 0.0;
	// setHDRMap("/home/zshore/projects/repos/akasha/cackle/hdr/skygrassHDR.exr");
	setHDRMap("/DPA/wookie/dpa/projects/zshore/products/hdr/image/0001/exr/none/skygrassHDR.exr");

	delete hdr_lights;
	hdr_lights = nullptr;
	n_hdr_lights = 0;
	uniforms->hdrlight_count = 0;
}

void CackleRenderer::initCubemap()
{
	camera = std::unique_ptr<Camera>(new Camera());
	uniforms = std::unique_ptr<Uniforms>(new Uniforms());

	// Round dpi to next multiple of 128
	if(dpi % 128)
		dpi = dpi + (128 - dpi % 128);

	// Round canvas_width and canvas_height to nearest even value
	if(canvas_width % 2)
		canvas_width = canvas_width + (2 - canvas_width % 2);

	if(canvas_height % 2)
		canvas_height = canvas_height + (2 - canvas_height % 2);

	img_width = std::floor(canvas_width * dpi);
	img_height = std::floor(canvas_height * dpi);

	// Cubemap is 6 widths wide and 1 height tall
	ull_img_size = static_cast<unsigned long long>(img_width) * 6ULL * static_cast<unsigned long long>(img_height) * 4ULL;
	image = std::unique_ptr<float[]>(new float[ull_img_size]);
	for(int i = 0; i < ull_img_size / 4ULL; i++)
	{
		image[i] = 0.0;
		image[i+1] = 0.0;
		image[i+2] = 0.0;
		image[i+3] = 1.0;
	}

	if(img_width * img_height < max_tile_width)
		tile_width = img_width * img_height;
	else
		tile_width = max_tile_width;

	tile_height = 1;
	tile = new DataObject<float4>(tile_width * tile_height);

	// Init HDR map
	n_hdr_lights = 1;
	uniforms->hdrlight_count = n_hdr_lights;
	uniforms->hdrmap_rotation = 0.0;
	setHDRMap("/DPA/wookie/dpa/projects/zshore/products/hdr/image/0001/exr/none/skygrassHDR.exr");
	
	delete hdr_lights;
	hdr_lights = nullptr;
	n_hdr_lights = 0;
	uniforms->hdrlight_count = 0;
}

// Camera Parameters

void CackleRenderer::setCameraEye(float x, float y, float z)
{
	camera->eye = make_float4(x, y, z, 0.0);
}

void CackleRenderer::setCameraAim(float x, float y, float z)
{
	camera->aim = make_float4(x, y, z, 0.0);
}

void CackleRenderer::setCameraView(float x, float y, float z)
{
	float len = sqrt(x*x + y*y + z*z);
	if(len < 0.000005) return;
	camera->view = make_float4(x/len, y/len, z/len, 0.0);
}

void CackleRenderer::setCameraUp(float x, float y, float z)
{
	float len = sqrt(x*x + y*y + z*z);
	if(len < 0.000005) return;
	camera->up = make_float4(x/len, y/len, z/len, 0.0);
}

void CackleRenderer::setCameraRight(float x, float y, float z)
{
	float len = sqrt(x*x + y*y + z*z);
	if(len < 0.000005) return;
	camera->right = make_float4(x/len, y/len, z/len, 0.0);
}

void CackleRenderer::setCameraFOV(float fov)
{
	camera->fov = fov;
}

// Fractal Parameters

void CackleRenderer::usePathTracing(bool upt)
{
	uniforms->use_path_tracing = upt;
}

void CackleRenderer::setMaxPathLength(int mpl)
{
	uniforms->max_path_length = mpl;
}

void CackleRenderer::setSampleCount(int sc)
{
	uniforms->samples = sc;
}

void CackleRenderer::setDither(float dither)
{
	uniforms->dither = dither;
}

void CackleRenderer::setFudgeFactor(float fudgefactor)
{
	uniforms->fudgefactor = fudgefactor;
}

void CackleRenderer::setFractalIterations(int fi)
{
	uniforms->fractal_iterations = fi;
}

void CackleRenderer::setRaymarchIterations(int ri)
{
	uniforms->raymarch_iterations = ri;
}

void CackleRenderer::setFLimit(float x, float y, float z)
{
	uniforms->flimit = make_float4(x, y, z, 0.0f);
}

void CackleRenderer::setJuliaC(float x, float y, float z)
{
   uniforms->julia_c = make_float4(x, y, z, 0.0f);
}

void CackleRenderer::setJuliaOffset(float x, float y, float z)
{
   uniforms->julia_offset = make_float4(x, y, z, 0.0f);
}

void CackleRenderer::setDEOffset(float deo)
{
   uniforms->de_offset = deo;
}

void CackleRenderer::setEpsilon(float eps)
{
	uniforms->epsilon = eps;
}

void CackleRenderer::setScale(float scale)
{
	uniforms->scale = scale;
}

void CackleRenderer::setMR2(float mr2)
{
	uniforms->mr2 = mr2;
}

void CackleRenderer::setFR2(float fr2)
{
	uniforms->fr2 = fr2;
}

void CackleRenderer::setFogLevel(float fog)
{
	uniforms->fog_level = fog;
}

void CackleRenderer::setFogColor(float r, float g, float b)
{
	uniforms->fog_color = make_float4(r, g, b, 0.0f);
}

void CackleRenderer::setGlowStrength(float glow)
{
	uniforms->glow_strength = glow;
}

void CackleRenderer::setSpecularExponent(float spec)
{
	uniforms->specular_exponent = spec;
}

void CackleRenderer::setGamma(float g)
{
	uniforms->gamma = g;
}

void CackleRenderer::setReflectivity(float refl)
{
	uniforms->reflectivity = refl;
}

void CackleRenderer::setAlbedo(float a)
{
	uniforms->albedo = a;
}

void CackleRenderer::setHDRMapRotation(float mr)
{
	uniforms->hdrmap_rotation = mr;
}

void CackleRenderer::setHDRLightCount(int lc)
{
	n_hdr_lights = lc;
	uniforms->hdrlight_count = lc;
}

void CackleRenderer::setColormapA(float r, float g, float b)
{
	uniforms->colormap_a = make_float4(r, g, b, 0.0);
}

void CackleRenderer::setColormapB(float r, float g, float b)
{
	uniforms->colormap_b = make_float4(r, g, b, 0.0);
}

void CackleRenderer::setColormapC(float r, float g, float b)
{
	uniforms->colormap_c = make_float4(r, g, b, 0.0);
}

void CackleRenderer::setColormapD(float r, float g, float b)
{
	uniforms->colormap_d = make_float4(r, g, b, 0.0);
}

void CackleRenderer::setColormap(int id)
{
	switch(id)
	{
		case 0:
			break;

		case 1:
			uniforms->colormap_a = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_b = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_c = make_float4(1.00f, 1.00f, 1.00f, 0.0f);
			uniforms->colormap_d = make_float4(0.00f, 0.33f, 0.67f, 0.0f);
			break;

		case 2:
			uniforms->colormap_a = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_b = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_c = make_float4(1.00f, 1.00f, 1.00f, 0.0f);
			uniforms->colormap_d = make_float4(0.00f, 0.10f, 0.20f, 0.0f);
			break;

		case 3:
			uniforms->colormap_a = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_b = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_c = make_float4(1.00f, 1.00f, 1.00f, 0.0f);
			uniforms->colormap_d = make_float4(0.30f, 0.20f, 0.20f, 0.0f);
			break;

		case 4:
			uniforms->colormap_a = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_b = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_c = make_float4(1.00f, 1.00f, 0.50f, 0.0f);
			uniforms->colormap_d = make_float4(0.80f, 0.90f, 0.30f, 0.0f);
			break;

		case 5:
			uniforms->colormap_a = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_b = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_c = make_float4(1.00f, 0.70f, 0.40f, 0.0f);
			uniforms->colormap_d = make_float4(0.00f, 0.15f, 0.20f, 0.0f);
			break;

		case 6:
			uniforms->colormap_a = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_b = make_float4(0.50f, 0.50f, 0.50f, 0.0f);
			uniforms->colormap_c = make_float4(2.00f, 1.00f, 0.00f, 0.0f);
			uniforms->colormap_d = make_float4(0.50f, 0.20f, 0.25f, 0.0f);
			break;

		case 7:
			uniforms->colormap_a = make_float4(0.80f, 0.50f, 0.40f, 0.0f);
			uniforms->colormap_b = make_float4(0.20f, 0.40f, 0.20f, 0.0f);
			uniforms->colormap_c = make_float4(2.00f, 1.00f, 1.00f, 0.0f);
			uniforms->colormap_d = make_float4(0.00f, 0.25f, 0.25f, 0.0f);
			break;

		case 8:
			uniforms->colormap_a = make_float4(0.10f, 0.30f, 0.20f, 0.0f);
			uniforms->colormap_b = make_float4(0.20f, 0.60f, 0.20f, 0.0f);
			uniforms->colormap_c = make_float4(1.00f, 1.50f, 1.00f, 0.0f);
			uniforms->colormap_d = make_float4(0.50f, 0.25f, 0.50f, 0.0f);
			break;

		default:
			;
	}

}

void CackleRenderer::setHDRMap(const char* path)
{
	hdr_map_path = std::string(path);

	if( hdr_map )
		delete hdr_map;

	ImageInput *in = ImageInput::create(hdr_map_path.c_str());
	if(!in)
	{
		std::cout << "ERROR: Failed to open " << hdr_map_path << " for input..." << std::endl;
		return;
	}

	ImageSpec spec;
	in->open(hdr_map_path.c_str(), spec);
	int w = spec.width;
	int h = spec.height;
	int d = spec.nchannels;

	float *map = new float[w * h * d];
	hdr_map = new DataObject<float4>(w * h);

	in->read_image(TypeDesc::FLOAT, map);
	in->close ();
	delete in;

	std::cout << "LOG: Read in HDR map of dimensions: " << w << "x" << h << "x" << d << std::endl;
	if( d == 3 )
	{
		int k = 0;
		for(int j = 0; j < h; ++j)
		{
			for(int i = 0; i < w; ++i)
			{
				int x = i + (h - j - 1) * w;
				hdr_map->getHostPointer()[k].x = map[x * 3];
				hdr_map->getHostPointer()[k].y = map[x * 3 + 1];
				hdr_map->getHostPointer()[k].z = map[x * 3 + 2];
				hdr_map->getHostPointer()[k].w = 1.0;
				k++;
			}
		}
	}
	else if( d == 4 )
	{
		int k = 0;
		for(int j = 0; j < h; ++j)
		{
			for(int i = 0; i < w; ++i)
			{
				int x = i + (h - j - 1) * w;
				hdr_map->getHostPointer()[k].x = map[x * 4];
				hdr_map->getHostPointer()[k].y = map[x * 4 + 1];
				hdr_map->getHostPointer()[k].z = map[x * 4 + 2];
				hdr_map->getHostPointer()[k].w = map[x * 4 + 3];
				k++;
			}
		}
	}

	uniforms->hdrmap_width = w;
	uniforms->hdrmap_height = h;

	hdr_map->updateDevice();

	// Find n_hdr_lights brightest pixels in hdr_map and make them lights
	if(hdr_lights)
		delete hdr_lights;
	hdr_lights = new DataObject<HDRPointLight>(n_hdr_lights);

	for(int t = 0; t < n_hdr_lights; ++t)
	{
		float max_L = -10.0;
		int px = 0;
		int py = 0;
		for(int j = 0; j < h; ++j)
		{
			for(int i = 0; i < w; ++i)
			{
				float luminance = 0.0;
				int ndx = i + (h - j - 1) * w;
				for(int k = 0; k < d; ++k)
					luminance += map[ndx * d + k];
				luminance /= float(d);

				if(luminance > max_L)
				{
					max_L = luminance;
					px = i;
					py = j;
				}
			}
		}

		float theta = CUDART_PI_F * (py / float(w));
		float phi = 2.0 * CUDART_PI_F * (px / float(h)) + uniforms->hdrmap_rotation * 2.0 * CUDART_PI_F;

		hdr_lights->getHostPointer()[t].p.x = 100.0 * std::sin(theta) * std::cos(phi);
		hdr_lights->getHostPointer()[t].p.y = 100.0 * std::sin(theta) * std::sin(phi);
		hdr_lights->getHostPointer()[t].p.z = 100.0 * std::cos(theta);
		hdr_lights->getHostPointer()[t].p.w = 0.0;

		hdr_lights->getHostPointer()[t].color = hdr_map->getHostPointer()[px + py * w];
		
		hdr_lights->getHostPointer()[t].falloff = 2.0;

		hdr_lights->getHostPointer()[t].px = px;
		hdr_lights->getHostPointer()[t].py = py;

		int ndx = px + (h - py - 1) * w;
		for(int k = 0; k < d; ++k)
			map[ndx * d + k] = -1.0;
	}
	hdr_lights->updateDevice();

	delete [] map;
}

void CackleRenderer::addPointLight(float x, float y, float z, float r, float g, float b, float falloff)
{
	if( hdr_lights == nullptr )
		hdr_lights = new DataObject<HDRPointLight>(1);
	else
		hdr_lights->extend(1);

	hdr_lights->getHostPointer()[n_hdr_lights].p       = make_float4( x, y, z, 0.0);
	hdr_lights->getHostPointer()[n_hdr_lights].color   = make_float4( r, g, b, 0.0);
	hdr_lights->getHostPointer()[n_hdr_lights].falloff = falloff;

	n_hdr_lights++;
	uniforms->hdrlight_count = n_hdr_lights;
	hdr_lights->updateDevice();
}

// Utility

void CackleRenderer::printInfo()
{
	std::cout << "---------- Info ----------\n"
	<< "Canvas Dimensions: " << canvas_width << "in. x " << canvas_height << "in. @ " << dpi << "dpi \n"
	<< "Image Dimensions: " << img_width << " x " << img_height << "\n"
	<< "Tile Dimensions: " << tile_width << " x " << tile_height << "\n"
	<< "--------------------------\n"
	<< "Camera\n" 
	<< "--------------------------\n"
	<< "\tEye: [" << camera->eye.x << ", " << camera->eye.y << ", " << camera->eye.z << "]\n"
	<< "\tAim: [" << camera->aim.x << ", " << camera->aim.y << ", " << camera->aim.z << "]\n"
	<< "\tView: [" << camera->view.x << ", " << camera->view.y << ", " << camera->view.z << "]\n"
	<< "\tUp: [" << camera->up.x << ", " << camera->up.y << ", " << camera->up.z << "]\n"
	<< "\tRight: [" << camera->right.x << ", " << camera->right.y << ", " << camera->right.z << "]\n"
	<< "\tFOV: " << camera->fov << "\n"
	<< "--------------------------\n" 
	<< "Render Settings\n" 
	<< "--------------------------\n"
	<< "\tSamples/Pixel: " << uniforms->samples * uniforms->samples << "\n"
	<< "\tUse Path Tracing: " << ((uniforms->use_path_tracing) ? "ON" : "OFF") << "\n"
	<< "\tMax Path Length: " << uniforms->max_path_length << "\n"
	<< "\tDither: " << uniforms->dither << "\n"
	<< "\tFudge Factor: " << uniforms->fudgefactor << "\n"
	<< "\tReflectivity: " << uniforms->reflectivity << "\n"
	<< "\tAlbedo: " << uniforms->albedo << "\n"
	<< "\tRaymarch Iterations: " << uniforms->raymarch_iterations << "\n"
	<< "--------------------------\n" 
	<< "Fractal\n" 
	<< "--------------------------\n"
	<< "\tFractal Iterations: " << uniforms->fractal_iterations << "\n"
	<< "\tFLimit: [" << uniforms->flimit.x << ", " << uniforms->flimit.y << ", " << uniforms->flimit.z << ", " << uniforms->flimit.w << "]\n"
	<< "\tEpsilon: " << uniforms->epsilon << "\n"
	<< "\tScale: " << uniforms->scale << "\n"
	<< "\tMR2: " << uniforms->mr2 << "\n"
	<< "\tFR2: " << uniforms->fr2 << "\n"
	
	<< "\tScalefactor: " << uniforms->scalefactor << "\n"
	<< "\tInv. Scalefactor: " << uniforms->inverse_scalefactor << "\n"
	
	<< "\tBV Radius: " << uniforms->bv_radius << "\n"
	<< "--------------------------\n"
	<< "Color\n"
	<< "--------------------------\n"

	<< "\tFog Level: " << uniforms->fog_level << "\n"
	<< "\tFog Color: [" << uniforms->fog_color.x << ", " << uniforms->fog_color.y << ", " << uniforms->fog_color.z << ", " << uniforms->fog_color.w << "]\n"

	<< "\tGlow Strength: " << uniforms->glow_strength << "\n"
	<< "\tSpecular Exponent: " << uniforms->specular_exponent << "\n"

	<< "\tA: [" << uniforms->colormap_a.x << ", " << uniforms->colormap_a.y << ", " << uniforms->colormap_a.z << ", " << uniforms->colormap_a.w << "]\n"
	<< "\tB: [" << uniforms->colormap_b.x << ", " << uniforms->colormap_b.y << ", " << uniforms->colormap_b.z << ", " << uniforms->colormap_b.w << "]\n"
	<< "\tC: [" << uniforms->colormap_c.x << ", " << uniforms->colormap_c.y << ", " << uniforms->colormap_c.z << ", " << uniforms->colormap_c.w << "]\n"
	<< "\tD: [" << uniforms->colormap_d.x << ", " << uniforms->colormap_d.y << ", " << uniforms->colormap_d.z << ", " << uniforms->colormap_d.w << "]\n"
	<< "--------------------------\n";

	std::cout << "Lighting\n"
	<< "--------------------------\n";
	for(int t = 0; t < n_hdr_lights; ++t)
	{
		std::cout << "\tPointLight " << t << " Info\n"
			<< "\t\tP: ["
			<< hdr_lights->getHostPointer()[t].p.x << ", "
			<< hdr_lights->getHostPointer()[t].p.y << ", "
			<< hdr_lights->getHostPointer()[t].p.z << "] " << std::endl
			<< "\t\tColor: ["
			<< hdr_lights->getHostPointer()[t].color.x << ", "
			<< hdr_lights->getHostPointer()[t].color.y << ", "
			<< hdr_lights->getHostPointer()[t].color.z << "] " << std::endl
			<< "\t\tFalloff: " << hdr_lights->getHostPointer()[t].falloff << std::endl;
		std::cout << std::endl;
	}
}

// Rendering

void CackleRenderer::renderTile(int tx, int ty)
{
	size_t tpb_x = 128;
	// size_t tpb_y = 16;

	size_t gx = (tile_width * tile_height) / tpb_x;
	// size_t gy = (tile_height) / tpb_y;

	if( gx * tpb_x < tile_width * tile_height ) gx++;
	// if( gy * tpb_y < tile_height ) gy++;

	util::ExecutionPolicy policy( tpb_x, gx );

	// std::cout 
	// 	<< "Rendering Tile: [" << tx << ", " << ty << "]" << std::endl
	// 	<< "\tThread block: (" << tpb_x << ")" << std::endl
	// 	<< "\t Grid Layout: (" << gx << ")" << std::endl;

	// First initialize a random state for each core
	curandState_t* states;
	cudaMalloc((void**)&states, tile_width * tile_height * sizeof(curandState_t));
	kernel_initRandomStates<<< policy.gridSize(), policy.blockSize()>>>(time(NULL), states);

	// Launch kernel
	kernel_render<<< policy.gridSize(), policy.blockSize() >>>(
		tile->getDevicePointer(), 
		tx, ty, tile_width, tile_height, 
		img_width, img_height, 
		*(camera.get()), *(uniforms.get()), 
		hdr_map->getDevicePointer(), hdr_lights->getDevicePointer(),
		states
	);

	tile->updateHost();
	cudaFree(states);
}

void CackleRenderer::renderImage(float* img)
{
	unsigned long long ull_iw = img_width;
	unsigned long long ull_ih = img_height;
	unsigned long long ull_tw = tile_width;

	unsigned long long nx = (ull_iw * ull_ih) / ull_tw;
	std::cout << "Rendering Image using " << nx << " tiles of size " << tile_width << "x" << tile_height << std::endl;

	lux::ProgressMeter *pm = new lux::ProgressMeter(nx, "Tiled Render");
	for(int i = 0; i < nx; ++i)
	{
		renderTile(i, 0);

		unsigned long long ull_i = i;
		for(unsigned long long j = 0; j < ull_tw; ++j)
		{
			unsigned long long ndx = ( j + ull_i * ull_tw ) * 4ULL;

			if( ndx >= ull_iw * ull_ih * 4ULL )
				break;

			img[ndx       ] = static_cast<float>(tile->getHostPointer()[j].x);
			img[ndx + 1ULL] = static_cast<float>(tile->getHostPointer()[j].y);
			img[ndx + 2ULL] = static_cast<float>(tile->getHostPointer()[j].z);
			img[ndx + 3ULL] = static_cast<float>(tile->getHostPointer()[j].w);
		}
		pm->update();
	}
	delete pm;

}

void CackleRenderer::render()
{
	// Calculate some extra parameters
	uniforms->scalefactor = 1.0f;
/*
	if(uniforms->scale < -1.0f)
		uniforms->scalefactor = 2.0f;
	else if(uniforms->scale > 1.0f)
		uniforms->scalefactor = 2.0f * (uniforms->scale + 1.0f) / (uniforms->scale - 1.0f);
*/

	uniforms->inverse_scalefactor = 1.0f / uniforms->scalefactor;
	float s = uniforms->scalefactor;
	uniforms->bv_radius = sqrt(3.0*s*s);

	std::cout << "Starting basic render with Cackle...\n\n";
	printInfo();
	
	renderImage(image.get());
}

void CackleRenderer::renderChunk(int cx, int n_chunks)
{
	// Calculate some extra parameters
	uniforms->scalefactor = 1.0f;
	uniforms->inverse_scalefactor = 1.0f / uniforms->scalefactor;
	float s = uniforms->scalefactor;
	uniforms->bv_radius = sqrt(3.0*s*s);

	std::cout << "Starting chunk render with Cackle...\n\n";
	printInfo();

	unsigned long long ull_iw = img_width;
	unsigned long long ull_ih = img_height;
	unsigned long long ull_tw = tile_width;
	unsigned long long ull_nc = n_chunks;

	unsigned long long cs = (ull_iw * ull_ih) / ull_nc;
	unsigned long long tpc = cs / ull_tw;

	std::cout << "Rendering chunk " << cx+1 << " of " << n_chunks << std::endl;
	std::cout << "Rendering chunk using " << tpc << " tiles of size " << tile_width << "x" << tile_height << std::endl;

	lux::ProgressMeter *pm = new lux::ProgressMeter(tpc, "Tiled Render");

	float* img = image.get();
	for(int i = 0; i < tpc; ++i)
	{
		renderTile(i + tpc * cx, 0);

		unsigned long long ull_i = i + tpc * cx;
		for(unsigned long long j = 0; j < ull_tw; ++j)
		{
			unsigned long long ndx = ( j + ull_i * ull_tw ) * 4ULL;

			if( ndx >= ull_iw * ull_ih * 4ULL )
				break;

			img[ndx       ] = static_cast<float>(tile->getHostPointer()[j].x);
			img[ndx + 1ULL] = static_cast<float>(tile->getHostPointer()[j].y);
			img[ndx + 2ULL] = static_cast<float>(tile->getHostPointer()[j].z);
			img[ndx + 3ULL] = static_cast<float>(tile->getHostPointer()[j].w);
		}
		pm->update();
	}

	delete pm;

}

void CackleRenderer::renderCubemap()
{
	// Calculate some extra parameters
	uniforms->scalefactor = 1.0f;
	if(uniforms->scale < -1.0f)
		uniforms->scalefactor = 2.0f;
	else if(uniforms->scale > 1.0f)
		uniforms->scalefactor = 2.0f * (uniforms->scale + 1.0f) / (uniforms->scale - 1.0f);

	uniforms->inverse_scalefactor = 1.0f / uniforms->scalefactor;
	float s = uniforms->scalefactor;
	uniforms->bv_radius = sqrt(3.0*s*s);

	std::cout << "Starting Cubemap render with Cackle...\n\n";
	printInfo();
	
	/*
		Setup Camera

	Cubemap Layout for NVIDIA Photoshop plugin:
	-----------------------------
	Right -- Left -- Front -- Back -- Top -- Bottom
	-----------------------------
	Each face is rendered separate and then copied into full image
	Render Order: Right, Left, Front, Back, Top, Bottom
	*/

	float* face = new float[img_width * img_height * 4];
	camera->fov = 90.0;

	// Right
	camera->view  = make_float4( 1.0, 0.0, 0.0, 0.0 );
	camera->up    = make_float4( 0.0, 0.0, 1.0, 0.0 );
	camera->right = make_float4( 0.0,-1.0, 0.0, 0.0 );

	std::cout << "\n--- Render Face --- : Right " << std::endl;
	renderImage(face);
	for(int j = 0; j < img_height; ++j)
		for(int i = 0; i < img_width; ++i)
		{
			int ii = i;
			int row = img_width * 6;

			int indx = (ii + j * row) * 4;
			int fndx = (i + j * img_width) * 4;

			for(int c = 0; c < 4; ++c)
				image[indx + c] = face[fndx + c];
		}

	// Left
	camera->view  = make_float4(-1.0, 0.0, 0.0, 0.0 );
	camera->up    = make_float4( 0.0, 0.0, 1.0, 0.0 );
	camera->right = make_float4( 0.0, 1.0, 0.0, 0.0 );

	std::cout << "\n--- Render Face --- : Left " << std::endl;
	renderImage(face);
	for(int j = 0; j < img_height; ++j)
		for(int i = 0; i < img_width; ++i)
		{
			int ii = i + img_width;
			int row = img_width * 6;

			int indx = (ii + j * row) * 4;
			int fndx = (i + j * img_width) * 4;

			for(int c = 0; c < 4; ++c)
				image[indx + c] = face[fndx + c];
		}

	// Front
	camera->view  = make_float4( 0.0, 0.0, 1.0, 0.0 );
	camera->up    = make_float4( 0.0,-1.0, 0.0, 0.0 );
	camera->right = make_float4( 1.0, 0.0, 0.0, 0.0 );

	std::cout << "\n--- Render Face --- : Front " << std::endl;
	renderImage(face);
	for(int j = 0; j < img_height; ++j)
		for(int i = 0; i < img_width; ++i)
		{
			int ii = i + 2 * img_width;
			int row = img_width * 6;

			int indx = (ii + j * row) * 4;
			int fndx = (i + j * img_width) * 4;

			for(int c = 0; c < 4; ++c)
				image[indx + c] = face[fndx + c];
		}

	// Back
	camera->view  = make_float4( 0.0, 0.0,-1.0, 0.0 );
	camera->up    = make_float4( 0.0, 1.0, 0.0, 0.0 );
	camera->right = make_float4( 1.0, 0.0, 0.0, 0.0 );

	std::cout << "\n--- Render Face --- : Back " << std::endl;
	renderImage(face);
	for(int j = 0; j < img_height; ++j)
		for(int i = 0; i < img_width; ++i)
		{
			int ii = i + 3 * img_width;
			int row = img_width * 6;

			int indx = (ii + j * row) * 4;
			int fndx = (i + j * img_width) * 4;

			for(int c = 0; c < 4; ++c)
				image[indx + c] = face[fndx + c];
		}

	// Top
	camera->view  = make_float4( 0.0, 1.0, 0.0, 0.0 );
	camera->up    = make_float4( 0.0, 0.0, 1.0, 0.0 );
	camera->right = make_float4( 1.0, 0.0, 0.0, 0.0 );

	std::cout << "\n--- Render Face --- : Top " << std::endl;
	renderImage(face);
	for(int j = 0; j < img_height; ++j)
		for(int i = 0; i < img_width; ++i)
		{
			int ii = i + 4 * img_width;
			int row = img_width * 6;

			int indx = (ii + j * row) * 4;
			int fndx = (i + j * img_width) * 4;

			for(int c = 0; c < 4; ++c)
				image[indx + c] = face[fndx + c];
		}

	// Bottom
	camera->view  = make_float4( 0.0,-1.0, 0.0, 0.0 );
	camera->up    = make_float4( 0.0, 0.0, 1.0, 0.0 );
	camera->right = make_float4(-1.0, 0.0, 0.0, 0.0 );

	std::cout << "\n--- Render Face --- : Bottom " << std::endl;
	renderImage(face);
	for(int j = 0; j < img_height; ++j)
		for(int i = 0; i < img_width; ++i)
		{
			int ii = i + 5 * img_width;
			int row = img_width * 6;

			int indx = (ii + j * row) * 4;
			int fndx = (i + j * img_width) * 4;

			for(int c = 0; c < 4; ++c)
				image[indx + c] = face[fndx + c];
		}

	delete [] face;

	std::cout << "\n--- Cubemap Render --- : Complete" << std::endl;
}

void CackleRenderer::outputImage(const char* filepath)
{
	// Convert image and write to file
	ImageOutput *out = ImageOutput::create(filepath);
	if(!out)
	{
		std::cout << "Unable to open file for writing: " << filepath << std::endl;
	} else
	{
		ImageSpec spec(img_width, img_height, 4, TypeDesc::FLOAT);

		spec.attribute("XResolution", static_cast<float>(dpi));
		spec.attribute("YResolution", static_cast<float>(dpi));
		spec.attribute("ResolutionUnit", std::string("in"));

		out->open(filepath, spec);
		out->write_image(TypeDesc::FLOAT, image.get());
		out->close();
		delete out;
	}

	std::cout << "Image written to " << filepath << std::endl;
}

void CackleRenderer::outputCubemap(const char* filepath)
{
	// Convert cubemap and write to file
	ImageOutput *out = ImageOutput::create(filepath);
	if(!out)
	{
		std::cout << "Unable to open file for writing: " << filepath << std::endl;
	} else
	{
		ImageSpec spec(img_width * 6, img_height, 4, TypeDesc::UINT16);

		spec.attribute("XResolution", static_cast<float>(dpi));
		spec.attribute("YResolution", static_cast<float>(dpi));
		spec.attribute("ResolutionUnit", std::string("in"));

		for(int i = 0; i < (img_width * 6) * img_height * 4; ++i)
			image[i] = powf(image[i], 0.45);

		out->open(filepath, spec);
		out->write_image(TypeDesc::FLOAT, image.get());
		out->close();
		delete out;
	}

	std::cout << "Cubemap written to " << filepath << std::endl;
}


}
