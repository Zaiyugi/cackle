#!/usr/bin/env python

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
rend = cackle.CackleRenderer(8, 6, 256, 4096)

rend.init()

rend.setFractalIterations(16)

rend.setRaymarchIterations(2500)
rend.setMaxPathLength(4)
rend.setSampleCount(25)

rend.setEpsilon(0.5e-5)

# rend.setScale(-1.8)
# rend.setMR2(0.27)

# rend.setScale(2.0)

rend.setScale(-1.7)
rend.setMR2(0.01)

rend.setDither(0.5);
rend.setFudgeFactor(0.95);

rend.setFogLevel(30.0)
rend.setFogColor(0.718, 0.847, 0.914)

rend.setGlowStrength(0.0)
rend.setSpecularExponent(50.0)
rend.setGamma(1.0)

rend.setColormap( 2 )
# rend.setHDRMap( "./hdr/skygrassHDR.exr" )

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

# rend.setCameraEye(  0.9, 0.8, 0.9 )
# rend.setCameraEye( -0.9, 0.8, 0.8 )
rend.setCameraEye( -0.875, 0.85, 1.0 )

# rend.setCameraView(  0.0,  0.0, -1.0 )
# rend.setCameraUp(    0.0,  1.0,  0.0 )
# rend.setCameraRight( 1.0,  0.0,  0.0 )

rend.setCameraView(  0.57, 0.03, -0.82 )
rend.setCameraUp(   -0.02, 1.00,  0.03 )
rend.setCameraRight( 0.82, 0.00,  0.57 )

# rend.setCameraView( -0.707, 0.00, -0.707 )
# rend.setCameraUp(    0.000, 1.00,  0.000 )
# rend.setCameraRight( 0.707, 0.00, -0.707 )

# rend.setCameraView( -1.0, -1.0, -1.0 )
# rend.setCameraUp(   -1.0,  1.0, -1.0 )
# rend.setCameraRight( 1.0,  0.0, -1.0 )

# rend.setCameraFOV(	 97.5)
rend.setCameraFOV(	 90.0)
# rend.setCameraFOV(	 82.5)

##############################################

rend.render()
# rend.outputImage("/DPA/wookie/zshore/fractals/posters/tiff/mandelbox_poster_22x14x2014.0001.tiff")
# rend.outputImage("/DPA/wookie/zshore/fractals/posters/tiff/mandelbox_poster_6x14x640_PT.0001.tiff")
rend.outputImage("/DPA/wookie/zshore/fractals/posters/tiff/mandelbox_poster_test_v2.0001.tiff")
