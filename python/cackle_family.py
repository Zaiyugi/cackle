#!/usr/bin/env python

import math
import sys
import cackle

# -- CackleRenderer:
# Render fractals
# (width in inches, height in inches, dpi)

#rend = cackle.CackleRenderer(20, 12, 512, 4096)
#rend = cackle.CackleRenderer(22, 12, 384, 4096)
rend = cackle.CackleRenderer(24, 14, 128, 4096)

rend.init()

##############################################

### The Glacier Wall ###

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

rend.setFractalIterations(20)
rend.setRaymarchIterations(30000)
rend.setFLimit(1.320056, 1.290057, 0.480084)
rend.setEpsilon(0.00015)
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
rend.addPointLight(0.26605, -2.5,  0.0, 0.00, 0.75, 0.75, 2.0);

rend.render()
rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/glacier_wall.0001.tiff")

##############################################

### Super Collider ###

rend.usePathTracing(False)
rend.setMaxPathLength(10)
rend.setSampleCount(3)

rend.setFractalIterations(20)
rend.setRaymarchIterations(30000)
rend.setFLimit(2.61001, 0.960068, 2.61001)
rend.setEpsilon(0.0001)
rend.setScale(-1)
rend.setMR2(0)
rend.setFR2(3)
rend.setCameraEye(0.19923, -0.000982925, -0.265496)
rend.setCameraAim(0.153914, -0.00710318, -0.20827)
rend.setCameraView(-0.618629, -0.0835508, 0.781228)
rend.setCameraUp(-0.0499868, 0.996684, 0.0642098)
rend.setCameraRight(-0.784004, 0.000670912, -0.620755)
rend.setCameraFOV(110)

# rend.render()
# rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/supercollider.0001.tiff")

##############################################

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

### Marble Lattice ###

rend.usePathTracing(False)
rend.setMaxPathLength(10)
rend.setSampleCount(3)

rend.setFractalIterations(16)
rend.setRaymarchIterations(30000)
rend.setFLimit(1.02007, 2.55001, 1.02007)
rend.setEpsilon(0.0001)
rend.setScale(1.032)
rend.setMR2(0)
rend.setFR2(1.44005)
rend.setCameraEye(0.755797, -4.6964, -0.851608)
rend.setCameraAim(0.398772, -4.85784, -0.116207)
rend.setCameraView(-0.42846, -0.193746, 0.882544)
rend.setCameraUp(-0.0724909, 0.984727, 0.158294)
rend.setCameraRight(-0.899915, 0.00384703, -0.436049)
rend.setCameraFOV(110)

# Light position for Marble Lattice
# rend.setCameraEye(-0.510272, -4.37148, -0.516767)

# rend.render()
# rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/marblelattice.0001.tiff")

##############################################

### Cauliflower ###

rend.usePathTracing(False)
rend.setMaxPathLength(10)
rend.setSampleCount(3)

rend.setFractalIterations(20)
rend.setRaymarchIterations(30000)
rend.setFLimit(1.02007, 2.28002, 1.02007)
rend.setEpsilon(0.0001)
rend.setScale(-1)
rend.setMR2(0)
rend.setFR2(3)
rend.setCameraEye(0.888791, -3.73809, 0.908055)
rend.setCameraAim(0.272751, -3.94193, 0.488644)
rend.setCameraView(-0.797323, -0.263835, -0.542832)
rend.setCameraUp(-0.20693, 0.966525, -0.15169)
rend.setCameraRight(0.564699, -0.00861793, -0.825252)
rend.setCameraFOV(110)

# rend.render()
# rend.outputImage("/DPA/wookie/zshore/fractals/posters/src/winter2017/cauliflower.0001.tiff")
