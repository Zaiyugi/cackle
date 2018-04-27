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

rend.setFractalIterations(16)
rend.setRaymarchIterations(30000)
rend.setFLimit(1.02007, 2.55001, 1.02007)
rend.setEpsilon(0.0002)
rend.setScale(1.032)
rend.setMR2(0)
rend.setFR2(1.44005)

rend.setCameraEye(0.755797, -4.6964, -0.851608)
# rend.setCameraAim(0.398772, -4.85784, -0.116207)

rend.setCameraView(-0.42846, -0.193746, 0.882544)
rend.setCameraUp(-0.0724909, 0.984727, 0.158294)
rend.setCameraRight(-0.899915, 0.00384703, -0.436049)

rend.setCameraFOV(110)

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
