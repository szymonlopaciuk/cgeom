import xtrack as xt
from matplotlib import pyplot as plt
from cpymad.madx import Madx
from cgeom.aperture import Aperture

# mad = Madx()
# mad.call('toy_ring_funky.seq')

lattice_file = 'toy_ring.seq'

# env = xt.load(lattice_file)
# env.set_particle_ref('proton', p0c=1.2e9)
# ring = env['ring']
#
# tw = ring.twiss4d()
#
# aperture_limit = Aperture.from_line_with_limit(ring, line_name='ring')
# aperture_limit.get_aperture_margin_mm(line_name="ring", element_name="mqf::0")  # return a vector at each s position
# aperture_limit.get_aperture_sigma(line_name="ring", element_name="mqf::0")  # return a vector at each s position
# aperture_limit.get_aperture_sigma_hv(line_name="ring", element_name="mqf::0")  # return a vector at each s position


mad = Madx(stdout=True)
mad.call(lattice_file)
env = xt.Environment.from_madx(lattice_file, enable_layout_data=True)
ring = env['ring']

aperture_meta = Aperture.from_line_with_aperture(ring, line_name='ring')
aperture_meta.get_aperture_margin_mm(line_name="ring", element_name="mqf")  # return a vector at each s position
# aperture_meta.get_aperture_sigma(line_name="ring", element_name="mqf")  # return a vector at each s position
# aperture_meta.get_aperture_sigma_hv(line_name="ring", element_name="mqf")  # return a vector at each s position

# for the MBRC lofting is needed
# look into ex_interpolate.py
