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

### Marble Lattice ###

rend.usePathTracing(True)
rend.setMaxPathLength(10)
rend.setSampleCount(8)

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

rend.setColormap( 2 )

### Mandelbox To Test ###
'''
rend.setFractalIterations(20)
rend.setRaymarchIterations(1024)
rend.setFLimit(1.02007, 1.02007, 1.02007)
rend.setJuliaC(0, 0, 0)
rend.setJuliaOffset(0, 0, 0)
rend.setEpsilon(0.0001)
rend.setDEOffset(0)
rend.setScale(1.172)
rend.setMR2(0.2)
rend.setFR2(1.05007)
rend.setCameraEye(1.38111, 2.30132, -0.277015)
rend.setCameraAim(0.66369, 1.73053, -0.543594)
rend.setCameraView(-0.751422, -0.597834, -0.279213)
rend.setCameraUp(-0.556829, 0.80195, -0.216372)
rend.setCameraRight(0.35327, -0.00711271, -0.935495)
rend.setCameraFOV(90)
'''

'''
rend.setFractalIterations(20)
rend.setRaymarchIterations(1024)
rend.setFLimit(1.02007, 1.02007, 1.02007)
rend.setJuliaC(0, 0, 0)
rend.setJuliaOffset(0, 0, 0)
rend.setEpsilon(0.0001)
rend.setDEOffset(0)
rend.setScale(1.0936)
rend.setMR2(0.2)
rend.setFR2(1.05007)
rend.setCameraEye(2.04982, 1.8818, -0.770839)
rend.setCameraAim(1.1409, 1.6837, -0.985708)
rend.setCameraView(-0.951996, -0.207496, -0.225052)
rend.setCameraUp(-0.200717, 0.978367, -0.0501038)
rend.setCameraRight(0.23058, -0.00252683, -0.97305)
rend.setCameraFOV(90)
'''

'''
rend.setFractalIterations(20)
rend.setRaymarchIterations(1024)
rend.setFLimit(1.02007, 1.02007, 1.02007)
rend.setJuliaC(0, 0, 0)
rend.setJuliaOffset(0, 0, 0)
rend.setEpsilon(0.0001)
rend.setDEOffset(0)
rend.setScale(1.0936)
rend.setMR2(0.2)
rend.setFR2(1.05007)
rend.setCameraEye(1.98613, 1.43201, -0.368153)
rend.setCameraAim(1.07539, 1.62632, -0.578739)
rend.setCameraView(-0.953902, 0.20352, -0.220567)
rend.setCameraUp(0.200209, 0.978367, 0.0520979)
rend.setCameraRight(0.226399, 0.00553683, -0.974019)
rend.setCameraFOV(90)
'''

###################

# Light position for Marble Lattice
# rend.setCameraEye(-0.510272, -4.37148, -0.516767)

fill_inten = 0.085
r17_inten = 0.05
r66_inten = 0.05
r357_inten = 0.2

# Roscolux R06: Pale Gold
rend.addPointLight( 0.00, -4.60, -0.00, 0.94 * fill_inten, 0.81 * fill_inten, 0.56 * fill_inten, 2.0);

# Roscolux R17: Light Flame
# rend.addPointLight( 0.00, -4.35,  0.51, 0.94 * r17_inten, 0.25 * r17_inten, 0.13 * r17_inten, 2.0);
# rend.addPointLight(-0.51, -4.35,  0.00, 0.94 * r17_inten, 0.25 * r17_inten, 0.13 * r17_inten, 2.0);
rend.addPointLight( 0.35, -4.85, -0.70, 0.94 * r17_inten, 0.25 * r17_inten, 0.13 * r17_inten, 2.0);

# Roscolux R66: Cool Blue
rend.addPointLight( 0.70, -4.85, -0.35, 0.00 * r66_inten, 0.75 * r66_inten, 0.75 * r66_inten, 2.0);

# Roscolux R357: Royal Lavender
rend.addPointLight(-0.51, -4.35,  0.00, 0.19 * r357_inten, 0.00 * r357_inten, 0.44 * r357_inten, 2.0);
rend.addPointLight( 0.00, -4.35,  0.51, 0.19 * r357_inten, 0.00 * r357_inten, 0.44 * r357_inten, 2.0);

# rend.render()
# rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/marblelattice.0001.tiff")

##############################################

# Queue setup

chunk = int(sys.argv[1])
total_chunks = int(sys.argv[2])
output = sys.argv[3]

rend.renderChunk(chunk, total_chunks)
rend.outputImage(output)
