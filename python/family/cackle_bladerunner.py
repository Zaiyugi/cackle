#!/usr/bin/env python

import math
import sys
import os
import cackle

# -- CackleRenderer:
# Render fractals
# (width in inches, height in inches, dpi)

rend = cackle.CackleRenderer(14, 24, 512, 4096)

rend.init()

##############################################

### Megacity ###

rend.usePathTracing(True)
rend.setMaxPathLength(5)
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

rend.setHDRLightCount(1)
hdr_path = os.path.abspath("/DPA/wookie/dpa/projects/zshore/products/hdr/image/0001/exr/none/sky0026.hdr")
rend.setHDRMap(hdr_path)
rend.setHDRMapRotation(0.5)

rend.setColormap( 3 )

rend.setFractalIterations(20)
rend.setRaymarchIterations(30000)

rend.setFLimit(1.80004, 3, 1.80004)
rend.setEpsilon(0.0001)
# rend.setScale(-2.024)
rend.setScale(-2.068)
rend.setMR2(1)
rend.setFR2(1.86004)

# rend.setCameraEye(3.00649, 5.86793, -2.3252)
rend.setCameraEye(3.00649, 5.8, -2.3)
# rend.setCameraAim(3.00059, 5.66917, -1.48739)
rend.setCameraView(0.0, 0.0, 1.0)
rend.setCameraUp(0.0, 1.0, 0.0)
rend.setCameraRight(-1.0, 0.0, 0.0)

# rend.setCameraView(-0.00685489, -0.230824, 0.972971)
# rend.setCameraUp(0.0312441, 0.968807, 0.245837)
# rend.setCameraRight(-0.999485, 0.0320886, 0.000570889)

# Roscolux R66: Cool Blue
# rend.addPointLight(3.00, 5.0, -3.0, 0.00, 0.75, 0.75, 2.0);
# rend.addPointLight(3.00, 5.5,  2.0, 0.00, 0.75, 0.75, 2.0);
# rend.addPointLight(3.00, 5.8,  0.0, 0.00, 0.75, 0.75, 2.0);
# rend.addPointLight(3.00649, 5.0, -2.1, 0.31, 0.75, 0.82, 2.0);

# Roscolux R70: Nile Blue
rend.addPointLight(3.00649, 4.2, -2.33, 0.06, 0.63, 0.69, 2.0);

rend.setCameraFOV(130)

# rend.render()
# rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/megacity.0001.tiff")

##############################################

# Queue setup

chunk = int(sys.argv[1])
total_chunks = int(sys.argv[2])
output = sys.argv[3]

rend.renderChunk(chunk, total_chunks)
rend.outputImage(output)
