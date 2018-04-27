#!/usr/bin/env python

import math
import sys
import cackle

# -- CackleRenderer:
# Render fractals
# (width in inches, height in inches, dpi)

# rend = cackle.CackleRenderer(14, 24, 128, 4096)
rend = cackle.CackleRenderer(24, 14, 512, 4096)

rend.init()

##############################################

### The Grid ###

rend.usePathTracing(True)
rend.setMaxPathLength(10)
rend.setSampleCount(9)

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

rend.setFractalIterations(20)
rend.setRaymarchIterations(30000)
rend.setFLimit(1.02007, 1.02007, 1.02007)
rend.setEpsilon(0.0001)
rend.setScale(-1.208)
rend.setMR2(0.08)
rend.setFR2(1.20006)

rend.setCameraEye(-0.00868217, -1.91486, -0.162987)
rend.setCameraAim(0.0350789, -1.93793, -0.0747575)
rend.setCameraView(0.432626, -0.228091, 0.872244)
rend.setCameraUp(0.108765, 0.973825, 0.199588)
rend.setCameraRight(-0.894938, 0.00852307, 0.44611)

rend.setCameraFOV(90)

prim_inten = 0.0003
top_inten = 0.001
low_inten = 0.005

# Roscolux R06: Pale Gold
rend.addPointLight( -0.0, -1.924, -0.12, 0.94 * prim_inten, 0.81 * prim_inten, 0.56 * prim_inten, 2.0);

# Roscolux R70: Nile Blue
rend.addPointLight( -0.00, -1.85, -0.00, 0.06 * low_inten, 0.63 * low_inten, 0.69 * low_inten, 2.0);
# rend.addPointLight(  0.00, -1.85, -0.24, 0.06 * low_inten, 0.63 * low_inten, 0.69 * low_inten, 2.0);

# Roscolux R17: Light Flame
rend.addPointLight(  0.12, -1.85, -0.12, 0.94 * low_inten, 0.25 * low_inten, 0.13 * low_inten, 2.0);
# rend.addPointLight( -0.12, -1.85, -0.12, 0.94 * low_inten, 0.25 * low_inten, 0.13 * low_inten, 2.0);

# Roscolux R357: Royal Lavender
rend.addPointLight(  0.00, -1.84, -0.12, 0.19 * top_inten, 0.00 * top_inten, 0.44 * top_inten, 2.0);
# rend.addPointLight( -0.00, -1.85, -0.00, 0.19 * lgt_inten, 0.00 * lgt_inten, 0.44 * lgt_inten, 2.0);

# rend.render()
# rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/thegrid.0001.tiff")

##############################################

# Queue setup

chunk = int(sys.argv[1])
total_chunks = int(sys.argv[2])
output = sys.argv[3]

rend.renderChunk(chunk, total_chunks)
rend.outputImage(output)
