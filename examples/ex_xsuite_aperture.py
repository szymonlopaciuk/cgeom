import xtrack as xt

from cgeom.aperture import Aperture

path = 'https://raw.githubusercontent.com/xsuite/xcoll/refs/heads/main/examples/machines/lhc_run3_b1.json'
# path = '../../xtrack/test_data/line_and_particle/line_lhc_no_errors.json'
lhc = xt.load(path)

aperture_limit = Aperture.from_line_with_limit(lhc.lhcb1, line_name='lhcb1')
aperture_limit.get_aperture_margin_mm(line_name="lhcb1", element_name="vmdqb.a1r1.a.b1")  # return a vector at each s position
# aperture_limit.get_aperture_sigma(line_name="lhcb1", element_name="vmdqb.a1r1.a.b1")  # return a vector at each s position
# aperture_limit.get_aperture_sigma_hv(line_name="lhcb1", element_name="vmdqb.a1r1.a.b1")  # return a vector at each s position


# lhc_with_metadata = xt.load('./lhc/lhc_aperture.json')
# aperture_meta = Aperture.from_line_with_aperture(lhc_with_metadata.b1, line_name='b1')
# aperture_meta.get_aperture_sigma(line_name="b1", element_name="mbrc.4r1", resolution=0.1)  # return a vector at each s position
# aperture_meta.get_aperture_sigma(line_name="b1", element_name="mb.a18r1.b1", resolution=0.1)  # return a vector at each s position

# for the MBRC lofting is needed
# look into ex_interpolate.py
