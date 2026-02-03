import numpy as np

import xtrack as xt

from cgeom.aperture import Aperture

lhc_with_metadata = xt.load('./lhc/lhc_aperture.json')
lhc_length = lhc_with_metadata.get_length()

aperture_model = Aperture.from_line_with_madx_metadata(lhc_with_metadata.b1, line_name='b1')

mqxfa_name = 'mqxfa.a1r1/b1'
mqxfa_sigmas = aperture_model.get_aperture_sigma(line_name="b1", element_name=mqxfa_name, resolution=0.1)



# aperture_meta.get_aperture_sigma(line_name="b1", element_name="mbrc.4r1", resolution=0.1)  # return a vector at each s position
# aperture_meta.get_aperture_sigma(line_name="b1", element_name="mb.a18r1.b1", resolution=0.1)  # return a vector at each s position

# for the MBRC lofting is needed
# look into ex_interpolate.py
