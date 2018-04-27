#!/usr/bin/env python

import cackle

# rend = cackle.CackleRenderer(30720, 17280, 480, 270) # Landscape
# rend = cackle.CackleRenderer(17280, 30720, 270, 480) # Portrait

# rend = cackle.CackleRenderer(15360, 8640, 480, 270)
# rend = cackle.CackleRenderer(7680, 4320, 480, 270)
# rend = cackle.CackleRenderer(3840, 2160, 480, 270)
rend = cackle.CackleRenderer(1920, 1080, 480, 270)

rend.init()

rend.setFractalIterations(10)
rend.setRaymarchIterations(4096)

rend.setFogLevel(0.1)
rend.setFogColor(0.0718, 0.0847, 0.0914)

rend.setEpsilon(0.5e-4)
# rend.setEpsilon(0.5e-3)
rend.setScale(1.05)
rend.setMR2(0.1)

rend.setDither(0.5);
rend.setFudgeFactor(0.75);

rend.setGlowStrength(0.1)
rend.setSpecularExponent(50.0)

rend.setColormap( 2 )

rend.setCameraEye(   0.0, 0.8, 0.6 )
rend.setCameraAim(   0.0, 0.8, 0.0 )
rend.setCameraView(  0.0, -0.3, -1.0 )
rend.setCameraUp(    0.0, 1.0,  0.0 )
rend.setCameraRight( 1.0, 0.0,  0.0 )
rend.setCameraFOV(	 65.0)

rend.render()
rend.outputImage("/DPA/wookie/zshore/fractals/GotGv2_PseudoKleinianMenger_hd.0001.exr")
