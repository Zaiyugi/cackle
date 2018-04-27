#!/usr/bin/env python

import cackle

# -- CackleRenderer:
# Render fractals
# (width in inches, height in inches, dpi, fraction of w/h for tile)
rend = cackle.CackleRenderer(16, 16, 256, 0.0625)

# rend.init()

# Try cubemap
rend.initCubemap()

rend.setFractalIterations(15)
rend.setRaymarchIterations(10000)

rend.setEpsilon(0.5e-7)
rend.setScale(-1.8)
rend.setMR2(0.27)

rend.setDither(0.5);
rend.setFudgeFactor(0.9);

rend.setFogLevel(0.0)
rend.setFogColor(0.718, 0.847, 0.914)

rend.setGlowStrength(0.0)
rend.setSpecularExponent(50.0)

rend.setColormap( 2 )

# rend.setCameraEye(   -0.56, 0.39, 1.47 )
# rend.setCameraAim(   -0.37, 0.39, 0.13 )
# rend.setCameraView(   0.14, 0.0, -0.99 )
# rend.setCameraUp(     0.00, 1.0, 0.00 )
# rend.setCameraRight(  0.99, 0.0, 0.14 )
# rend.setCameraFOV(	  94.5)

# rend.setCameraEye(  -0.72, 0.73, 0.78 )
# rend.setCameraAim(  -0.38, 0.49, 0.13 )
# rend.setCameraView(  0.26, -0.12, -0.96 )
# rend.setCameraUp(    0.03, 0.99, -0.12 )
# rend.setCameraRight( 0.96, 0.00, 0.27 )
# rend.setCameraFOV(	 85)

# rend.setCameraEye(  -0.86,  0.00, -0.83 )
# rend.setCameraAim(  -0.44,  0.05, -0.10 )
# rend.setCameraView( -0.26, -0.03, -0.96 )
# rend.setCameraUp(   -0.01,  1.00, -0.03 )
# rend.setCameraRight( 0.97,  0.00, -0.26 )
# rend.setCameraFOV(	 113)

##############################################

### Fave ###
# rend.setCameraEye(  -0.78, 0.595,  0.70 )
# rend.setCameraAim(  -0.34, 0.305,  0.03 )
# rend.setCameraView(  0.57, 0.03, -0.82 )
# rend.setCameraUp(   -0.02, 1.00,  0.03 )
# rend.setCameraRight( 0.82, 0.00,  0.57 )
# rend.setCameraFOV(	 105.5)

##############################################

### Anim ###
# rend.setCameraEye(  0.75, 0.75, 0.75 )
rend.setCameraEye(  0.74, 0.625, 0.74 )
# rend.setCameraEye(  -0.58, 0.75, -0.58 )
# rend.setCameraAim(  -0.39, 0.68, -0.24 )
# rend.setCameraView( -0.70, 0.10, -0.70 )
rend.setCameraView( -0.707, 0.00, -0.707 )
rend.setCameraUp(    0.000, 1.00,  0.000 )
rend.setCameraRight( 0.707, 0.00, -0.707 )
# rend.setCameraFOV(	 97.5)
rend.setCameraFOV(	 90.0)

##############################################

# rend.setCameraEye(  -0.58, 0.75, -0.58 )
# rend.setCameraAim(  -0.39, 0.68, -0.24 )
# rend.setCameraView( -0.71, 0.10, -0.70 )
# rend.setCameraUp(    0.07, 0.99,  0.07 )
# rend.setCameraRight( 0.70, 0.00, -0.71 )
# rend.setCameraFOV(	 97.5)

# rend.setCameraEye(  -0.00, 0.94, -0.00 )
# rend.setCameraAim(  -0.35, 0.95, 0.17 )
# rend.setCameraView(  0.26, -0.09, 0.96 )
# rend.setCameraUp(    0.02, 1.00, 0.09 )
# rend.setCameraRight( -0.96, 0.00, 0.26 )
# rend.setCameraFOV(	 115.5)

# rend.setCameraEye(  -0.00, 0.88, 0.00 )
# rend.setCameraAim(  -0.35, 0.95, 0.17 )
# rend.setCameraView(  0.64, -0.63, -0.43 )
# rend.setCameraUp(    0.52, 0.77, -0.36 )
# rend.setCameraRight( 0.56, 0.00, 0.83 )
# rend.setCameraFOV(	 90)

# rend.setCameraEye(  -0.13, 0.80, -0.14 )
# rend.setCameraAim(  -0.35, 0.95, 0.17 )
# rend.setCameraView(  0.67, -0.24, 0.70 )
# rend.setCameraUp(    0.16, 0.97, 0.17 )
# rend.setCameraRight( -0.72, 0.00, 0.69 )
# rend.setCameraFOV(	 105)

# rend.setCameraEye(  -0.86, 1.01, 0.24 )
# rend.setCameraAim(  -0.36, 0.89, 0.18 )
# rend.setCameraView(  0.81, 0.02, -0.59 )
# rend.setCameraUp(    -0.02, 1.00, 0.01 )
# rend.setCameraRight( 0.59, 0.00, 0.81 )
# rend.setCameraFOV(	 104)

# rend.setCameraEye(  -0.49, 0.49, 0.82 )
# rend.setCameraAim(  -0.60, 0.59, 0.10 )
# rend.setCameraView(  -0.73, 0.58, 0.36 )
# rend.setCameraUp(    0.52, 0.82, -0.25 )
# rend.setCameraRight( -0.44, 0.00, -0.90 )
# rend.setCameraFOV(	 127)

# rend.setCameraEye(  -0.88, 0.49, -0.89 )
# rend.setCameraAim(  -0.87, 0.79, 0.10 )
# rend.setCameraView(  0.71, -0.03, 0.71)
# rend.setCameraUp(    0.02, 1.00, 0.02 )
# rend.setCameraRight( -0.71, 0.00, 0.71 )
# rend.setCameraFOV(	 104)

# rend.setCameraEye(  -0.88, 0.41, -0.89 )
# rend.setCameraAim(  -0.87, 0.71, 0.10 )
# rend.setCameraView(  0.78, -0.01, 0.62)
# rend.setCameraUp(    0.01, 1.00, 0.01 )
# rend.setCameraRight( -0.62, 0.00, 0.78 )
# rend.setCameraFOV(	 104)

# rend.setCameraEye(  -0.38, 0.44, 0.72 )
# rend.setCameraAim(  -0.72, 0.34, 0.13 )
# rend.setCameraView(  0.52, 0.63, -0.57)
# rend.setCameraUp(    -0.42, 0.78, 0.47 )
# rend.setCameraRight( 0.74, 0.00, 0.67 )
# rend.setCameraFOV(	 90)

rend.renderCubemap()
rend.outputCubemap("/DPA/wookie/zshore/fractals/posters/cubemaps/mandelbox_cubemap_4096-uint16.0001.tiff")

# cam.eye   = cackle.make_float4(-0.53, 0.73, -0.51)
# cam.aim   = cackle.make_float4(-0.39, 0.69, -0.24)
# cam.view  = cackle.make_float4(0.95, -0.17, 0.27)
# cam.up    = cackle.make_float4(0.16, 0.99, 0.05)
# cam.right = cackle.make_float4(-0.28, 0.00, 0.96)
# cam.fov   = 88.5

# rend.render(cam, params)
# rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0002.exr")

# cam.eye   = cackle.make_float4(-0.62, 0.71, -0.61)
# cam.aim   = cackle.make_float4(-0.39, 0.69, -0.24)
# cam.view  = cackle.make_float4(-0.66, -0.33, -0.67)
# cam.up    = cackle.make_float4(-0.23, 0.94, -0.24)
# cam.right = cackle.make_float4(0.71, 0.00, -0.70)
# cam.fov   = 88.5

# rend.render(cam, params)
# rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0003.exr")

# cam.eye   = cackle.make_float4(-0.60, 0.72, -0.59)
# cam.aim   = cackle.make_float4(-0.39, 0.69, -0.24)
# cam.view  = cackle.make_float4(0.29, 0.91, 0.28)
# cam.up    = cackle.make_float4(-0.66, 0.41, -0.63)
# cam.right = cackle.make_float4(-0.69, 0.00, 0.72)
# cam.fov   = 88.5

# rend.render(cam, params)
# rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0004.exr")

# cam.eye   = cackle.make_float4(-0.58, 0.75, -0.58)
# cam.aim   = cackle.make_float4(-0.39, 0.68, -0.24)
# cam.view  = cackle.make_float4(-0.71, 0.10, -0.70)
# cam.up    = cackle.make_float4(0.07, 0.99, 0.07)
# cam.right = cackle.make_float4(0.70, 0.00, -0.71)
# cam.fov   = 97.5

# rend.render(cam, params)
# rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0005.exr")

# cam.eye   = cackle.make_float4(-0.59, 0.87, -0.53)
# cam.aim   = cackle.make_float4(-0.39, 0.68, -0.24)
# cam.view  = cackle.make_float4(0.95, -0.16, -0.28)
# cam.up    = cackle.make_float4(0.15, 0.99, -0.05)
# cam.right = cackle.make_float4(0.29, 0.00, 0.96)
# cam.fov   = 97.5

# rend.render(cam, params)
# rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0006.exr")

# cam.eye   = cackle.make_float4(-0.47, 0.82, -0.47)
# cam.aim   = cackle.make_float4(-0.39, 0.68, -0.24)
# cam.view  = cackle.make_float4(0.80, 0.00, 0.59)
# cam.up    = cackle.make_float4(-0.00, 1.00, -0.00)
# cam.right = cackle.make_float4(-0.59, 0.00, 0.80)
# cam.fov   = 97.5

# rend.render(cam, params)
# rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0007.exr")

# cam.eye   = cackle.make_float4(-0.34, 0.88, -0.34)
# cam.aim   = cackle.make_float4(-0.39, 0.67, -0.24)
# cam.view  = cackle.make_float4(-0.75, -0.14, -0.65)
# cam.up    = cackle.make_float4(-0.11, 0.99, -0.09)
# cam.right = cackle.make_float4(0.66, 0.00, -0.75)
# cam.fov   = 97.5

# rend.render(cam, params)
# rend.outputImage("/DPA/wookie/zshore/mandelbox_wedge_uhd_H_16k.0008.exr")

