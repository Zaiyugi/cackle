#!/usr/bin/env python

import math
import sys
import cackle

# -- CackleRenderer:
# Render fractals
# (width in inches, height in inches, dpi)

rend = cackle.CackleRenderer(24, 14, 512, 4096)

rend.init()

##############################################

### The Precipice ###

rend.usePathTracing(True)
rend.setMaxPathLength(5)
rend.setSampleCount(8)

rend.setDither(0.5);
rend.setFudgeFactor(0.5);

rend.setFogLevel(0.0)
rend.setFogColor(0.00718, 0.00847, 0.00457)

rend.setGlowStrength(0.0)
rend.setSpecularExponent(65.0)
rend.setGamma(1.0)

rend.setReflectivity(0.5)
rend.setAlbedo(1.0)

rend.setHDRMapRotation(0.0)

rend.setColormap( 5 )

rend.setFractalIterations(20)
rend.setRaymarchIterations(30000)
rend.setFLimit(1.320056, 1.290057, 0.480084)
rend.setEpsilon(0.0002)
rend.setScale(1.048)
rend.setMR2(0.04)
rend.setFR2(1.560048)

eye = [0.179252, -1.77982, -0.463859]
aim = [0.66605, -1.78172, -0.433351]
view = [0.963061, -0.00391984, 0.369254]
up = [-0.0395816, 0.999108, -0.0147128]
right = [-0.369241, 0.00351548, 0.963066]

rend.setCameraEye(eye[0], eye[1], eye[2])
rend.setCameraAim(aim[0], aim[1], aim[2])
rend.setCameraView(view[0], view[1], view[2])
rend.setCameraUp(up[0], up[1], up[2])
rend.setCameraRight(right[0], right[1], right[2])
rend.setCameraFOV(110)

# Roscolux R06: Pale Gold
# rend.addPointLight(0.5, -0.75, 0.000, 0.94, 0.81, 0.56, 1.5);

# Roscolux R17: Light Flame
rend.addPointLight(0.5, -0.75, 0.000, 0.94, 0.25, 0.13, 2.0);

# Roscolux R66: Cool Blue
rend.addPointLight(0.26605, -2.3,  1.5, 0.00, 0.75, 0.75, 2.0);
rend.addPointLight(0.26605, -2.3, -1.5, 0.00, 0.75, 0.75, 2.0);
# rend.addPointLight(0.26605, -2.5,  0.0, 0.00, 0.75, 0.75, 2.0);

rend.render()
rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/precipice.0001.tiff")

##############################################

# Queue setup

# chunk = int(sys.argv[1])
# total_chunks = int(sys.argv[2])
# output = sys.argv[3]

# rend.renderChunk(chunk, total_chunks)
# rend.outputImage(output)
