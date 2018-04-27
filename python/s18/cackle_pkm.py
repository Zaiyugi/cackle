#!/usr/bin/env python

import math
import sys
import cackle

# -- CackleRenderer:
# Render fractals
# (width in inches, height in inches, dpi)

rend = cackle.CackleRenderer(10, 4, 512, 4096)

rend.init()

##############################################

rend.usePathTracing(True)
rend.setMaxPathLength(5)
rend.setSampleCount(6)

rend.setDither(0.5);
rend.setFudgeFactor(0.9);

rend.setFogLevel(0.0)
rend.setFogColor(0.00718, 0.00847, 0.00457)

rend.setGlowStrength(0.0)
rend.setSpecularExponent(65.0)
rend.setGamma(1.0)

rend.setReflectivity(0.5)
rend.setAlbedo(0.5)

rend.setHDRMapRotation(0.0)

rend.setColormap( 2 )

### PKM To Test ###
'''
rend.setFractalIterations(20)
rend.setRaymarchIterations(30000)

rend.setFLimit(1.38005, 1.80004, 2.01003)
rend.setJuliaC(0.376, 2, 0.436)
rend.setJuliaOffset(1, 0.984, 1)
rend.setEpsilon(0.0001)
rend.setDEOffset(0)
rend.setScale(3)
rend.setMR2(0.1)
rend.setFR2(0)

rend.setCameraEye(4.78426, 21.6613, -12.8741)
rend.setCameraAim(4.00728, 21.2098, -10.8658)
rend.setCameraView(-0.353139, -0.205222, 0.912785)
rend.setCameraUp(-0.029321, 0.975132, 0.219678)
rend.setCameraRight(-0.935223, 0.0508159, -0.350395)
rend.setCameraFOV(90)
'''

### World Caliber ###

rend.setFractalIterations(20)
rend.setRaymarchIterations(50000)

rend.setFLimit(1.38005, 1.80004, 2.01003)
rend.setJuliaC(0.376, 2, 0.436)
rend.setJuliaOffset(1, 0.984, 1)
rend.setEpsilon(0.0001)
rend.setDEOffset(0)
rend.setScale(3)
rend.setMR2(0.1)
rend.setFR2(0)

rend.setCameraEye(1.34118, 14.4316, -10.648)
rend.setCameraAim(3.48951, 14.1706, -10.2512)
rend.setCameraView(0.976426, -0.118583, 0.180359)
rend.setCameraUp(0.132321, 0.991184, -0.00669254)
rend.setCameraRight(-0.177986, 0.0304018, 0.983563)
rend.setCameraFOV(90)

###################

# Roscolux R17: Light Flame
r17_inten = 0.1
rend.addPointLight( 1.5, 14.40, -10.8, 0.94 * r17_inten, 0.25 * r17_inten, 0.13 * r17_inten, 2.0);
# rend.addPointLight( 1.34, 14.40, -10.625, 0.94 * r17_inten, 0.25 * r17_inten, 0.13 * r17_inten, 2.0);

# Roscolux R66: Cool Blue
r66_inten = 0.1
rend.addPointLight( 1.45, 14.40, -10.475, 0.00 * r66_inten, 0.75 * r66_inten, 0.75 * r66_inten, 2.0);
# rend.addPointLight( 1.34, 14.40, -10.625, 0.00 * r66_inten, 0.75 * r66_inten, 0.75 * r66_inten, 2.0);

# Roscolux R14: Medium Straw
r14_inten = 0.3
rend.addPointLight( 1.68, 14.40, -10.625, 1.00 * r14_inten, 0.67 * r14_inten, 0.19 * r14_inten, 2.0);
# rend.addPointLight( 1.34, 14.40, -10.625, 1.00 * r14_inten, 0.67 * r14_inten, 0.19 * r14_inten, 2.0);

# Roscolux R357: Royal Lavender
# r357_inten = 0.1
# rend.addPointLight(-0.51, -4.35,  0.00, 0.19 * r357_inten, 0.00 * r357_inten, 0.44 * r357_inten, 2.0);
# rend.addPointLight( 1.55, 14.40, -10.475, 0.19 * r357_inten, 0.00 * r357_inten, 0.44 * r357_inten, 2.0);

##############################################

# Queue setup

chunk = int(sys.argv[1])
total_chunks = int(sys.argv[2])
#output = sys.argv[3]

rend.renderChunk(chunk, total_chunks)
rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/spring2018/world_caliber.0001.tiff")
#rend.outputImage(output)
