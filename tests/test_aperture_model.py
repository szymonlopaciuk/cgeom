import xtrack as xt
import xobjects as xo
from cpymad.madx import Madx

from cgeom.aperture import Aperture
from cgeom.structures import Ellipse, Rectangle, RectEllipse
from itertools import zip_longest


TOY_RING_SEQUENCE = """
    ! Toy Ring, 4 arcs

    l_arc = 3;  ! length of the arc
    l_quad = 0.3;  ! length of the quads
    l_drift = 1;  ! length of the straight section drifts

    qf = 0.1;  ! qf strength
    qd = -0.7;  ! qd strength
    angle_arc = pi / 2;  ! arcs 90°

    mb: sbend, angle = angle_arc, l = l_arc, apertype=circle, aperture={0.1}, aper_offset={0.003, 0};
    mqf: quadrupole, k1 = qf, l = l_quad, apertype=rectangle, aperture={0.08, 0.04};
    mqd: quadrupole, k1 = qd, l = l_quad, apertype=ellipse, aperture={0.04, 0.08};
    ds: drift, l = l_drift;
    ap_ds: marker, apertype=rectellipse, aperture={0.022, 0.01715, 0.022, 0.022}, aper_tol={9e-4, 8e-4, 5e-4};
    dsa: line = (ap_ds, ds, ap_ds);

    ss_f: line = (dsa, mqf, dsa);
    ss_d: line = (dsa, mqd, dsa);

    ring: line = (ss_f, mb, ss_d, mb, ss_f, mb, ss_d, mb);

    beam, particle=proton, pc=1.2e9;
    use, period=ring;
"""


def test_aperture_from_line_with_aperture_type_bounds():
    mad = Madx(stdout=None)
    mad.input(TOY_RING_SEQUENCE)
    env = xt.Environment.from_madx(madx=mad, enable_layout_data=True)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_madx_metadata(ring, line_name='ring')
    type_bounds = aperture_model._type_bounds()
    type_name_bounds = [(a, b, aperture_model.type_name_for_position(c) if c else None) for a, b, c in type_bounds]
    table_rows = ring.get_table().cols['s_start', 's_end', 'name', 'element_type'].rows[:-1].rows

    for type_bound, table_row in zip_longest(type_name_bounds, table_rows):
        type_start = type_bound[0]
        type_end = type_bound[1]
        type_name = type_bound[2]

        element_start = table_row.s_start
        element_end = table_row.s_end
        element_name = table_row.name

        xo.assert_allclose(element_start, type_start, atol=1e-6)
        xo.assert_allclose(element_end, type_end, atol=1e-6)

        if table_row.element_type == 'Drift':  # MAD-X won't allow apertures on drifts, so these shouldn't have bounds
            assert type_name is None
            continue

        assert element_name.startswith(type_name)


def test_aperture_from_line_with_associated_apertures_type_bounds():
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_associated_apertures(ring, line_name='ring')
    type_bounds = aperture_model._type_bounds()
    type_name_bounds = [(a, b, aperture_model.type_name_for_position(c) if c else None) for a, b, c in type_bounds]
    table_rows = ring.get_table().cols['s_start', 's_end', 'name', 'element_type'].rows[:-1].rows

    for type_bound, table_row in zip_longest(type_name_bounds, table_rows):
        type_start = type_bound[0]
        type_end = type_bound[1]
        type_name = type_bound[2]

        element_start = table_row.s_start
        element_end = table_row.s_end
        element_name = table_row.name

        xo.assert_allclose(element_start, type_start, atol=1e-6)
        xo.assert_allclose(element_end, type_end, atol=1e-6)

        if table_row.element_type == 'Drift':  # MAD-X won't allow apertures on drifts, so these shouldn't have bounds
            assert type_name is None
            continue

        # element names in survey have ::N at the end, we make the check disregarding the suffix:
        prototype_name, suffix = element_name.split('::')
        _ = int(suffix)
        assert type_name.startswith(prototype_name)


def test_aperture_from_line_with_limits_type_bounds():
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=True)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_limits(ring, line_name='ring')
    type_bounds = aperture_model._type_bounds()
    type_name_bounds_only_limits = [(a, b, aperture_model.type_name_for_position(c)) for a, b, c in type_bounds if c]

    bounds_from_table = []
    for row in ring.get_table().rows:
        if row.element_type.startswith('Limit'):
            bounds_from_table.append((row.s_start, row.s_end, row.name))

    for type_bound, table_bound in zip_longest(type_name_bounds_only_limits, bounds_from_table):
        type_start = type_bound[0]
        type_end = type_bound[1]
        type_name = type_bound[2]

        element_start = table_bound[0]
        element_end = table_bound[1]
        element_name = table_bound[2]

        xo.assert_allclose(element_start, type_start, atol=1e-6)
        xo.assert_allclose(element_end, type_end, atol=1e-6)
        assert element_name.startswith(type_name)


def test_aperture_find_type_positions_perfect_overlap():
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_associated_apertures(ring, line_name='ring')

    mqf0, = aperture_model._find_type_positions(1, 1.3, 'ring')
    assert mqf0.survey_reference_name == 'mqf::0'

    mqf0_name = aperture_model.type_name_for_position(mqf0)
    assert mqf0_name == 'mqf_aper'

    mqf0_type = aperture_model.type_for_position(mqf0)
    mqf0_profile_names = [aperture_model.profile_name_for_position(pos) for pos in mqf0_type.positions]
    assert mqf0_profile_names == ['mqf_aper', 'mqf_aper']

    mqf0_prof_pos0, mqf0_prof_pos1 = mqf0_type.positions
    assert mqf0_prof_pos0.s_position == 0.
    assert mqf0_prof_pos1.s_position == 0.3
    assert mqf0_prof_pos0.shift_x == mqf0_prof_pos0.shift_y == mqf0_prof_pos1.shift_x == mqf0_prof_pos1.shift_y == 0.

    mqf0_profile_start, mqf0_profile_end = [aperture_model.profile_for_position(pos) for pos in mqf0_type.positions]
    assert isinstance(mqf0_profile_start, Rectangle)
    assert mqf0_profile_start.half_width == mqf0_profile_end.half_width == 0.08
    assert mqf0_profile_start.half_height == mqf0_profile_end.half_height == 0.04

def test_aperture_find_type_positions_partially_spanning_multiple_types():
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_associated_apertures(ring, line_name='ring')

    overlapping = aperture_model._find_type_positions(8, 11.8, 'ring')
    mb1, ap_ds8, ap_ds9, mqf1 = overlapping

    # Check the bend
    assert mb1.survey_reference_name == 'mb::1'

    mb1_name = aperture_model.type_name_for_position(mb1)
    assert mb1_name == 'mb_aper'

    mb1_type = aperture_model.type_for_position(mb1)
    xo.assert_allclose(mb1_type.curvature, ring['mb'].h, atol=1e-6)
    mb1_profile_names = [aperture_model.profile_name_for_position(pos) for pos in mb1_type.positions]
    assert mb1_profile_names == ['mb_aper', 'mb_aper']

    mb1_prof_pos0, mb1_prof_pos1 = mb1_type.positions
    assert mb1_prof_pos0.s_position == 0.
    assert mb1_prof_pos1.s_position == 3.
    assert mb1_prof_pos0.shift_x == mb1_prof_pos0.shift_y == mb1_prof_pos1.shift_x == mb1_prof_pos1.shift_y == 0.

    mb1_profile_start, mb1_profile_end = [aperture_model.profile_for_position(pos) for pos in mb1_type.positions]
    assert isinstance(mb1_profile_start, Ellipse)
    assert mb1_profile_start.half_major == mb1_profile_end.half_major == 0.1
    assert mb1_profile_start.half_minor == mb1_profile_end.half_minor == 0.1

    # Check the mqf
    mqf1_name = aperture_model.type_name_for_position(mqf1)
    assert mqf1_name == 'mqf_aper'

    mqf1_type = aperture_model.type_for_position(mqf1)
    mqf1_profile_names = [aperture_model.profile_name_for_position(pos) for pos in mqf1_type.positions]
    assert mqf1_profile_names == ['mqf_aper', 'mqf_aper']

    mqf1_prof_pos0, mqf1_prof_pos1 = mqf1_type.positions
    assert mqf1_prof_pos0.s_position == 0.
    assert mqf1_prof_pos1.s_position == 0.3
    assert mqf1_prof_pos0.shift_x == mqf1_prof_pos0.shift_y == mqf1_prof_pos1.shift_x == mqf1_prof_pos1.shift_y == 0.

    mqf1_profile_start, mqf1_profile_end = [aperture_model.profile_for_position(pos) for pos in mqf1_type.positions]
    assert isinstance(mqf1_profile_start, Rectangle)
    assert mqf1_profile_start.half_width == mqf1_profile_end.half_width == 0.08
    assert mqf1_profile_start.half_height == mqf1_profile_end.half_height == 0.04

    # Check the ap_ds
    ap_ds8_name = aperture_model.type_name_for_position(ap_ds8)
    ap_ds9_name = aperture_model.type_name_for_position(ap_ds8)
    assert ap_ds8_name == ap_ds9_name == 'ap_ds_aper'

    ap_ds8_type = aperture_model.type_for_position(ap_ds8)
    assert ap_ds8.type_index == ap_ds9.type_index

    ap_ds8_profile_names = [aperture_model.profile_name_for_position(pos) for pos in ap_ds8_type.positions]
    assert ap_ds8_profile_names == ['ap_ds_aper']

    ap_ds8_prof_pos0, = ap_ds8_type.positions
    assert ap_ds8_prof_pos0.s_position == 0.
    assert ap_ds8_prof_pos0.shift_x == ap_ds8_prof_pos0.shift_y

    ap_ds8_profile_start = aperture_model.profile_for_position(ap_ds8_prof_pos0)
    assert isinstance(ap_ds8_profile_start, RectEllipse)
    assert ap_ds8_profile_start.half_major == 0.022
    assert ap_ds8_profile_start.half_minor == 0.022
    assert ap_ds8_profile_start.half_width == 0.022
    assert ap_ds8_profile_start.half_height == 0.01715
