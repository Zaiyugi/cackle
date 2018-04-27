#!/usr/bin/env python

import cackle

params = cackle.Uniforms()

params.fractal_iterations = 15;
params.raymarch_iterations = 4096;
params.fog_level = 0.1;
params.epsilon = 0.5e-7;
params.scale = -1.8;
params.mr2 = 0.27;
params.glow_strength = 1.0;
params.specular_exponent = 150.0;

params.colormap_a = cackle.make_float4(0.5, 0.5, 0.5);
params.colormap_b = cackle.make_float4(0.5, 0.5, 0.5);
params.colormap_c = cackle.make_float4(1.0, 1.0, 1.0);
params.colormap_d = cackle.make_float4(0.0, 0.1, 0.2);

# rend = cackle.CackleRenderer(30720, 17280, 480, 270) # Landscape
# rend = cackle.CackleRenderer(17280, 30720, 270, 480) # Portrait

# rend = cackle.CackleRenderer(15360, 8640, 480, 270)
# rend = cackle.CackleRenderer(7680, 4320, 480, 270)
rend = cackle.CackleRenderer(3840, 2160, 480, 270)
# rend = cackle.CackleRenderer(1920, 1080, 480, 270)

rend.init()

cam = cackle.Camera()

# cam.eye = cackle.make_float4(-0.56, 0.39, 1.47)
# cam.aim = cackle.make_float4(-0.37, 0.39, 0.13)
# cam.view = cackle.make_float4(0.14, 0.0, -0.99)
# cam.up = cackle.make_float4(0.00, 1.0, 0.00)
# cam.right = cackle.make_float4(0.99, 0.0, 0.14)
# cam.fov = 94.5

# cam.eye = cackle.make_float4(-0.72, 0.73, 0.78)
# cam.aim = cackle.make_float4(-0.38, 0.49, 0.13)
# cam.view = cackle.make_float4(0.26, -0.12, -0.96)
# cam.up = cackle.make_float4(0.03, 0.99, -0.12)
# cam.right = cackle.make_float4(0.96, 0.00, 0.27)
# cam.fov = 85

# cam.eye   = cackle.make_float4(-0.86,  0.00, -0.83)
# cam.aim   = cackle.make_float4(-0.44,  0.05, -0.10)
# cam.view  = cackle.make_float4(-0.26, -0.03, -0.96)
# cam.up    = cackle.make_float4(-0.01,  1.00, -0.03)
# cam.right = cackle.make_float4( 0.97,  0.00, -0.26)
# cam.fov   = 113

# cam.eye   = cackle.make_float4(-0.78, 0.595,  0.70)
# cam.aim   = cackle.make_float4(-0.34, 0.305,  0.03)
# cam.view  = cackle.make_float4( 0.57, 0.03, -0.82)
# cam.up    = cackle.make_float4(-0.02, 1.00,  0.03)
# cam.right = cackle.make_float4( 0.82, 0.00,  0.57)
# cam.fov   = 105.5

### New cases
cam.eye   = cackle.make_float4(-0.43, 0.75, -0.47)
cam.aim   = cackle.make_float4(-0.39, 0.69, -0.24)
cam.view  = cackle.make_float4( 0.56, -0.04, 0.83)
cam.up    = cackle.make_float4( 0.02, 1.00, 0.03)
cam.right = cackle.make_float4(-0.83, 0.00, 0.56)
cam.fov   = 82.5

rend.render(cam, params)
rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0001.exr")

cam.eye   = cackle.make_float4(-0.53, 0.73, -0.51)
cam.aim   = cackle.make_float4(-0.39, 0.69, -0.24)
cam.view  = cackle.make_float4(0.95, -0.17, 0.27)
cam.up    = cackle.make_float4(0.16, 0.99, 0.05)
cam.right = cackle.make_float4(-0.28, 0.00, 0.96)
cam.fov   = 88.5

rend.render(cam, params)
rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0002.exr")

cam.eye   = cackle.make_float4(-0.62, 0.71, -0.61)
cam.aim   = cackle.make_float4(-0.39, 0.69, -0.24)
cam.view  = cackle.make_float4(-0.66, -0.33, -0.67)
cam.up    = cackle.make_float4(-0.23, 0.94, -0.24)
cam.right = cackle.make_float4(0.71, 0.00, -0.70)
cam.fov   = 88.5

rend.render(cam, params)
rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0003.exr")

cam.eye   = cackle.make_float4(-0.60, 0.72, -0.59)
cam.aim   = cackle.make_float4(-0.39, 0.69, -0.24)
cam.view  = cackle.make_float4(0.29, 0.91, 0.28)
cam.up    = cackle.make_float4(-0.66, 0.41, -0.63)
cam.right = cackle.make_float4(-0.69, 0.00, 0.72)
cam.fov   = 88.5

rend.render(cam, params)
rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0004.exr")

cam.eye   = cackle.make_float4(-0.58, 0.75, -0.58)
cam.aim   = cackle.make_float4(-0.39, 0.68, -0.24)
cam.view  = cackle.make_float4(-0.71, 0.10, -0.70)
cam.up    = cackle.make_float4(0.07, 0.99, 0.07)
cam.right = cackle.make_float4(0.70, 0.00, -0.71)
cam.fov   = 97.5

rend.render(cam, params)
rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0005.exr")

cam.eye   = cackle.make_float4(-0.59, 0.87, -0.53)
cam.aim   = cackle.make_float4(-0.39, 0.68, -0.24)
cam.view  = cackle.make_float4(0.95, -0.16, -0.28)
cam.up    = cackle.make_float4(0.15, 0.99, -0.05)
cam.right = cackle.make_float4(0.29, 0.00, 0.96)
cam.fov   = 97.5

rend.render(cam, params)
rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0006.exr")

cam.eye   = cackle.make_float4(-0.47, 0.82, -0.47)
cam.aim   = cackle.make_float4(-0.39, 0.68, -0.24)
cam.view  = cackle.make_float4(0.80, 0.00, 0.59)
cam.up    = cackle.make_float4(-0.00, 1.00, -0.00)
cam.right = cackle.make_float4(-0.59, 0.00, 0.80)
cam.fov   = 97.5

rend.render(cam, params)
rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0007.exr")

cam.eye   = cackle.make_float4(-0.34, 0.88, -0.34)
cam.aim   = cackle.make_float4(-0.39, 0.67, -0.24)
cam.view  = cackle.make_float4(-0.75, -0.14, -0.65)
cam.up    = cackle.make_float4(-0.11, 0.99, -0.09)
cam.right = cackle.make_float4(0.66, 0.00, -0.75)
cam.fov   = 97.5

rend.render(cam, params)
rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0008.exr")

