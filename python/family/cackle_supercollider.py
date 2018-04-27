#!/usr/bin/env python

import math
import sys
import cackle

# -- CackleRenderer:
# Render fractals
# (width in inches, height in inches, dpi)

# rend = cackle.CackleRenderer(14, 24, 128, 4096)
rend = cackle.CackleRenderer(8, 12, 128, 4096)

rend.init()

##############################################

### Super Collider ###

rend.usePathTracing(True)
rend.setMaxPathLength(2)
rend.setSampleCount(4)

rend.setDither(0.2);
rend.setFudgeFactor(0.5);

rend.setFogLevel(0.0)
rend.setFogColor(0.00718, 0.00847, 0.00457)

rend.setGlowStrength(0.0)
rend.setSpecularExponent(65.0)
rend.setGamma(1.0)

rend.setReflectivity(0.85)
rend.setAlbedo(1.0)

rend.setHDRMapRotation(0.0)

rend.setColormap( 8 )

rend.setFractalIterations(20)
rend.setRaymarchIterations(30000)
rend.setFLimit(2.61001, 0.960068, 2.61001)
rend.setEpsilon(0.00002)
rend.setScale(-1)
rend.setMR2(0)
rend.setFR2(3)

# rend.setCameraEye(0.19923, -0.000982925, -0.265496)
# rend.setCameraAim(0.153914, -0.00710318, -0.20827)
# rend.setCameraView(-0.618629, -0.0835508, 0.781228)
# rend.setCameraUp(-0.0499868, 0.996684, 0.0642098)
# rend.setCameraRight(-0.784004, 0.000670912, -0.620755)

rend.setCameraEye(   0.19923,  0.0, -0.265496)
rend.setCameraView( -0.618629, 0.0,  0.781228)
rend.setCameraUp(    0.0,      1.0,  0.0)
rend.setCameraRight(-0.784004, 0.0, -0.620755)

rend.setCameraFOV(110)

# Roscolux R06: Pale Gold
# rend.addPointLight(0.5, -0.75, 0.000, 0.94, 0.81, 0.56, 1.5);

# Roscolux R06: Pale Gold
rend.addPointLight( 0.1, 0.0,  0.1, 0.94, 0.81, 0.56, 2.0);
# rend.addPointLight( 0.1, -1.5,  0.1, 0.94, 0.81, 0.56, 2.0);

rend.render()
rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/supercollider.0001.tiff")

##############################################
