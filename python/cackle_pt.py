#!/usr/bin/env python

import math
import sys
import cackle

# rend = cackle.CackleRenderer(30720, 17280, 480, 270) # Landscape
# rend = cackle.CackleRenderer(17280, 30720, 270, 480) # Portrait

# rend = cackle.CackleRenderer(30720, 17280, 240, 135)
# rend = cackle.CackleRenderer(15360, 8640, 240, 135)

# rend = cackle.CackleRenderer(7680, 4320, 240, 135)
# rend = cackle.CackleRenderer(3840, 2160, 240, 135)

# -- CackleRenderer:
# Render fractals
# (width in inches, height in inches, dpi)
# rend = cackle.CackleRenderer(20, 16, 2048)
# rend = cackle.CackleRenderer(22, 14, 1024, 524288)

# rend = cackle.CackleRenderer(16, 12, 320, 262144)
# rend = cackle.CackleRenderer(6, 14, 640, 32768)

# rend = cackle.CackleRenderer(10, 24, 512, 4096)

rend = cackle.CackleRenderer(20, 12, 512, 4096)
#rend = cackle.CackleRenderer(22, 12, 384, 4096)
#rend = cackle.CackleRenderer(6, 4, 512, 4096)

rend.init()

rend.usePathTracing(True)

rend.setFractalIterations(20)

rend.setRaymarchIterations(30000)
rend.setMaxPathLength(10)
rend.setSampleCount(6)

rend.setEpsilon(0.5e-5)
#rend.setEpsilon(0.0001)

# rend.setScale(-1.8)
# rend.setMR2(0.27)

rend.setScale(1.24)
#rend.setScale(-1.0)

# rend.setScale(-1.525)
rend.setMR2(0.23)

rend.setDither(0.5);
rend.setFudgeFactor(0.9);

rend.setFogLevel(0.0)
rend.setFogColor(0.00718, 0.00847, 0.00457)

rend.setGlowStrength(0.0)
rend.setSpecularExponent(65.0)
rend.setGamma(1.0)

rend.setReflectivity(0.5)
rend.setAlbedo(1.0)

rend.setHDRMapRotation(0.0)
# rend.setHDRMap( "./hdr/skygrassHDR.exr" )

#rend.setColormap( 1 )
rend.setColormap( 2 )
#rend.setColormap( 5 )

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
# rend.setCameraEye(  0.806, 0.806, 0.806 )
# rend.setCameraEye(  0.81, 0.77, 0.81 )

# rend.setCameraEye(  0.8, 0.8, 0.8 )

# rend.setCameraEye(  0.8, 0.8, 0.8 )
# rend.setCameraEye(  0.74, 0.625, 0.74 )

# rend.setCameraEye(  -0.58, 0.75, -0.58 )
# rend.setCameraAim(  -0.39, 0.68, -0.24 )
# rend.setCameraView( -0.70, 0.10, -0.70 )

# rend.setCameraView( -0.707, 0.00, -0.707 )
# rend.setCameraUp(    0.000, 1.00,  0.000 )
# rend.setCameraRight( 0.707, 0.00, -0.707 )

# rend.setCameraView( -1.0, -1.0, -1.0 )
# rend.setCameraUp(   -1.0,  1.0, -1.0 )
# rend.setCameraRight( 1.0,  0.0, -1.0 )

# rend.setCameraFOV(	 97.5)
# rend.setCameraFOV(	 90.0)
# rend.setCameraFOV(	 82.5)

##############################################

### Path Tracing ###

# rend.setCameraEye(  0.0, 0.0, 0.0 )
# rend.setCameraEye(  0.42, 0.705, 0.83 )
# rend.setCameraEye( -0.9, 0.7, 0.9 )

# rend.setCameraEye( -0.875, 0.85, 0.95 )
# rend.setCameraEye(  -0.83, 0.71, 0.56 )

# rend.setCameraView(  0.0,  0.0, -1.0 )
# rend.setCameraUp(    0.0,  1.0,  0.0 )
# rend.setCameraRight( 1.0,  0.0,  0.0 )

# rend.setCameraView(  0.57, 0.03, -0.82 )
# rend.setCameraUp(   -0.02, 1.00,  0.03 )
# rend.setCameraRight( 0.82, 0.00,  0.57 )

# rend.setCameraView( -0.707, 0.00, -0.707 )
# rend.setCameraUp(    0.000, 1.00,  0.000 )
# rend.setCameraRight( 0.707, 0.00, -0.707 )

# rend.setCameraView( -1.0, -1.0, -1.0 )
# rend.setCameraUp(   -1.0,  1.0, -1.0 )
# rend.setCameraRight( 1.0,  0.0, -1.0 )

# rend.setCameraFOV(	 97.5)
# rend.setCameraFOV(	 90.0)
# rend.setCameraFOV(	 82.5)

### Juliabox ###

# rend.setCameraEye(  -1.0, -1.0,  2.5 )

# rend.setCameraView(  0.0,  0.0, -1.0 )
# rend.setCameraUp(    0.0,  1.0,  0.0 )
# rend.setCameraRight( 1.0,  0.0,  0.0 )

# rend.setCameraFOV( 90.0 )

##############################################

# Render this next!
#rend.setCameraEye(-0.372663, 2.14436, 0.824742)
#rend.setCameraEye(-0.35, 2.1, 0.77)
#rend.setCameraEye(-0.25, 2.05, 0.25)
# rend.setCameraAim(-0.4582, 2.04508, -0.166624)
# rend.setCameraView(-0.0855376, -0.0992891, -0.991375)
# rend.setCameraUp(-0.0125894, 0.995343, -0.0955734)
# rend.setCameraRight(0.996252, 0.00430575, -0.0863896)
#rend.setCameraView( -1.0,  0.0, -1.0)
#rend.setCameraUp(    0.0,  1.0,  0.0)
#rend.setCameraRight( 1.0,  0.0, -1.0)

#rend.setCameraFOV(120)
rend.setCameraFOV(110)

#rend.setCameraEye(0.276565, 2.18453, 0.29495)
rend.setCameraEye(0.29495, 1.5, 0.29495)

#rend.setCameraAim(0.0154454, 2.26106, -0.150062)
rend.setCameraView(-0.500603, 0.146731, -0.853151)
rend.setCameraUp(0.074532, 0.989092, 0.127055)
rend.setCameraRight(0.862488, 1.71587e-05, -0.506078)

#rend.setCameraFOV(65)

##############################################

rend.render()
rend.outputImage("/DPA/wookie/zshore/fractals/posters/pathtracing/mandelbox_test.0001.tiff")

### Render Sequence ###
# for i in range(1, total_frames+1):

# 	t = i / float(total_frames)

# 	z = t * (6.0 - -6.0) - 6.0
# 	a_t = t * math.pi * 0.5;

# 	z = -z
# 	rend.setCameraEye( -1.0, -1.0, z )

# 	rend.setCameraUp(    math.cos(a_t),  math.sin(a_t),  0.0 )

# 	a_t += math.pi * 0.5;
# 	rend.setCameraRight( math.cos(a_t),  math.sin(a_t),  0.0 )

# 	frame = str(i).zfill(4);

# 	rend.render()
# 	rend.outputImage("/DPA/wookie/zshore/fractals/posters/pathtracing/anim/camera_dolly."+frame+".tiff")
