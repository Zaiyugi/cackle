#!/usr/bin/env python

import cackle

cam = cackle.Camera()
params = cackle.Uniforms()

rend = cackle.CackleRenderer(30720, 17280, 480, 270)
# rend = cackle.CackleRenderer(15360, 8640, 480, 270)
# rend = cackle.CackleRenderer(7680, 4320, 480, 270)
# rend = cackle.CackleRenderer(3840, 2160, 480, 270)
# rend = cackle.CackleRenderer(1920, 1080)

cam.eye = cackle.make_float4(-0.56, 0.39, 1.47)
cam.aim = cackle.make_float4(-0.37, 0.39, 0.13)
cam.view = cackle.make_float4(0.14, 0.0, -0.99)
cam.up = cackle.make_float4(0.00, 1.0, 0.00)
cam.right = cackle.make_float4(0.99, 0.0, 0.14)
cam.fov = 104.5

params.fractal_iterations = 13;
params.raymarch_iterations = 2048;
params.fog_level = 0.1;
params.epsilon = 1.0e-5;
params.scale = 2.0;
params.mr2 = 0.1;
params.glow_strength = 0.25;

params.colormap_a = cackle.make_float4(0.5, 0.5, 0.5);
params.colormap_b = cackle.make_float4(0.5, 0.5, 0.5);
params.colormap_c = cackle.make_float4(1.0, 1.0, 1.0);
params.colormap_d = cackle.make_float4(0.0, 0.1, 0.2);

rend.init()
rend.setCamera(cam)
rend.setUniforms(params)

rend.render()

rend.outputImage("/DPA/wookie/zshore/mandelbox_uhd_32k.0001.exr")
