#!/usr/bin/env python

import math
import sys
import cackle

# -- CackleRenderer:
# Render fractals
# (width in inches, height in inches, dpi)

rend = cackle.CackleRenderer(24, 14, 128, 4096)

rend.init()

##############################################

rend.usePathTracing(True)
rend.setMaxPathLength(10)
rend.setSampleCount(4)

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

# Roscolux R06: Pale Gold
# rend.addPointLight(0.5, -0.75, 0.000, 0.94, 0.81, 0.56, 1.5);

# Roscolux R17: Light Flame
rend.addPointLight(0.5, -0.75, 0.000, 0.94, 0.25, 0.13, 2.0);

# Roscolux R66: Cool Blue
rend.addPointLight(0.26605, -2.3,  1.5, 0.00, 0.75, 0.75, 2.0);
rend.addPointLight(0.26605, -2.3, -1.5, 0.00, 0.75, 0.75, 2.0);
rend.addPointLight(0.26605, -2.5,  0.0, 0.00, 0.75, 0.75, 2.0);

### Cliff Diving ###

rend.usePathTracing(False)
rend.setMaxPathLength(10)
rend.setSampleCount(3)

rend.setFractalIterations(19)
rend.setRaymarchIterations(30000)
rend.setFLimit(1.77004, 1.56005, 1.56005)
rend.setEpsilon(0.0001)
rend.setScale(1.176)
rend.setMR2(1)
rend.setFR2(3)
rend.setCameraEye(3.57787, -2.86119, -0.0187142)
rend.setCameraAim(3.60166, -2.8679, -0.030647)
rend.setCameraView(0.866798, -0.244237, -0.434753)
rend.setCameraUp(0.235803, 0.966525, -0.101128)
rend.setCameraRight(0.444933, -0.0148602, 0.895441)
rend.setCameraFOV(110)

# rend.render()
# rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/cliffdiving.0001.tiff")

##############################################

# Queue setup

# chunk = int(sys.argv[1])
# total_chunks = int(sys.argv[2])
# output = sys.argv[3]

# rend.renderChunk(chunk, total_chunks)
# rend.outputImage(output)
