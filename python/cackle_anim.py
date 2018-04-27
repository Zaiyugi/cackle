#!/usr/bin/env python

import sys

import cackle

# rend = cackle.CackleRenderer(30720, 17280, 480, 270) # Landscape
# rend = cackle.CackleRenderer(17280, 30720, 270, 480) # Portrait

rend = cackle.CackleRenderer(15360, 8640, 480, 270)
# rend = cackle.CackleRenderer(7680, 4320, 480, 270)
# rend = cackle.CackleRenderer(3840, 2160, 480, 270)
# rend = cackle.CackleRenderer(1920, 1080, 480, 270)

rend.init()

rend.setFractalIterations(15)
rend.setRaymarchIterations(4096)

rend.setFogLevel(175.0)
rend.setFogColor(0.0718, 0.0847, 0.0914)

rend.setEpsilon(0.5e-7)
rend.setScale(1.8)
rend.setMR2(0.27)
rend.setGlowStrength(5.0)
rend.setSpecularExponent(50.0)

rend.setColormap( 2 )

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

rend.setCameraEye(  -0.82, 0.36, 0.39 )
rend.setCameraAim(  -0.46, 0.40, 0.08 )
rend.setCameraView(  0.75, 0.08, -0.65)
rend.setCameraUp(    -0.06, 1.00, 0.05 )
rend.setCameraRight( 0.65, -0.00, 0.76 )
rend.setCameraFOV(	 94.5)

frames = int(sys.argv[1])
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])
for i in range(start_frame, end_frame+1):
	f = i / float(frames)
	# f = f * 1.0 + 1.001
	# rend.setScale( f )

	frame = str(i).zfill(4)
	print("Render frame: " + frame)

	rend.render()
	
	print("Write frame " + frame + " to disk")
	rend.outputImage("/DPA/wookie/zshore/fractals/anim/mandelbox_wedge_negativeScale_16k."+frame+".exr")
