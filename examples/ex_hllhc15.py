import numpy as np
import inspect
import xtrack as xt
import matplotlib.pyplot as plt

from cgeom import clib, Path2D
from cgeom.beam_aperture import Aperture, LimitTypes

# env = xt.load('flatcc_opt_flatvh_75_180_1500_240_0/saved_b1.madx')
# env.set_particle_ref('proton', energy0=7e9)
# b1 = env.lhcb1
#
# sv_xt = b1.survey()
# tw_xt = b1.twiss4d()

sv = xt.survey.SurveyTable.from_tfs('flatcc_opt_flatvh_75_180_1500_240_0/survey_ir1b1.tfs')
sv['X'] = sv['x']
del sv['x']
sv['Y'] = sv['y']
del sv['y']
sv['Z'] = sv['z']
del sv['z']

tw = xt.TwissTable.from_tfs('flatcc_opt_flatvh_75_180_1500_240_0/twiss_ir1b1.tfs')

ap = xt.Table.from_tfs('flatcc_opt_flatvh_75_180_1500_240_0/ap_ir1b1.tfs')

name_to_match = "mqxfa.a1r1"
name_matches = lambda s: s.lower().startswith(name_to_match)


def make_octagon(half_width, half_height, angle_0, angle_1):
    x = 0.5 * half_width * (np.tan(angle_0) + 1)
    diag = np.sqrt(2) * x
    return Path2D.from_octagon(half_width, half_height, diag)


def make_shape(apertype, *args):
    match apertype:
        case 'octagon':
            return make_octagon(*args)
        case 'circle':
            return Path2D.from_ellipse(args[0], args[0])
        case 'ellipse':
            return Path2D.from_ellipse(args[0], args[1])
        case 'rectellipse':
            return Path2D.from_rectellipse(*args)
        case 'racetrack':
            return Path2D.from_racetrack(*args)
        case _:
            raise ValueError(f'Unsupported apertype {apertype}')

fig, ax = plt.subplots()
ax.set_aspect('equal')

nemitt_x = 2.5e-6
nemitt_y = 2.5e-6


def plot_halo_sigmas(row, ap, sig_x, sig_y, hr, hx, hy, **kwargs):
    tmp = np.sqrt(2 * (hr - hx) * (hr - hy))
    sh = hr - hy + tmp
    sv = hr - hx + tmp
    sr = hx + hy - hr - tmp

    d_arc = ap.dqf * ap.paras_dx
    dx = ap.beta_beating * d_arc * np.sqrt(row.betx / ap.betaqfx) * ap.twiss_deltap
    dy = ap.beta_beating * d_arc * np.sqrt(row.bety / ap.betaqfx) * ap.twiss_deltap
    co = ap.co_radius
    rtolx = row.rtol + co + dx
    rtoly = row.rtol + co + dy

    bh = row.xtol + sh * sig_x
    bv = row.ytol + sv * sig_y
    brx = rtolx + sr * sig_x
    bry = rtoly + sr * sig_y

    # Plot the beam halo
    beam_re = Path2D.from_racetrack(
        halfhside=bh + brx,
        halfvside=bv + bry,
        rx=brx,
        ry=bry,
    )
    print(f'--PYTHON @ {row.name}--')
    print(f' * s = {row.s}, h = {bh}, v = {bv}, rx = {brx}, ry = {bry}, x = {row.x}, y = {row.y}, hr = {hr}, hx = {hx}, hy = {hy}')
    print(f' * xtol = {row.xtol}, ytol = {row.ytol}, rxtol = {rtolx}, rytol = {rtoly}, sh = {sh}, sv = {sv}, sr = {sr}, sig_x = {sig_x}, sig_y = {sig_y}, n1 = {hr}')
    print(f'  * betx = {row.betx}, bety = {row.bety}, dx = {row.dx}, dy = {row.dy}')
    beam_points = beam_re.get_points(ds_min=None)
    off_x = ap.twiss_deltap * row.dx
    off_y = ap.twiss_deltap * row.dy
    ax.plot(row.x + beam_points['x'] + off_x, row.y + beam_points['y'] + off_y, **kwargs)
    ax.plot(row.x + beam_points['x'] - off_x, row.y + beam_points['y'] - off_y, **kwargs)


for row in ap.rows:
    if not name_matches(row.name):
        continue
    apertype = row.apertype.lower()
    args = [row.aper_1, row.aper_2, row.aper_3, row.aper_4]
    path = make_shape(apertype, *args)

    path.plot(ax=ax, color='k')

    sig_x = np.sqrt(row.betx * ap.exn / ap.gamma / ap.beta) * ap.beta_beating
    sig_y = np.sqrt(row.bety * ap.eyn / ap.gamma / ap.beta) * ap.beta_beating

    n1 = row.n1
    hr = ap.halo_r
    hx = ap.halo_h
    hy = ap.halo_v

    # Halos at the specified halo_{r, h, v} and also at n1
    #plot_halo_sigmas(row, ap, sig_x, sig_y, hr, hx, hy, c='g', label=fr'beam with halo = (x = {hr}$\sigma$, y = {hx}$\sigma$, r = {hy}$\sigma$))')
    plot_halo_sigmas(row, ap, sig_x, sig_y, n1, n1, n1, c='r', label=fr'beam with halo = (x = n1$\sigma$, y = n1$\sigma$, r = n1$\sigma$))')

    # Same but use cgeom
    beam_data = clib.G2DBeamData(
        emitx_norm=ap.exn,
        emity_norm=ap.eyn,
        delta_rms=ap.twiss_deltap,
        tol_co=ap.co_radius,
        tol_disp=ap.paras_dx,
        tol_disp_ref_dx=ap.betaqfx,
        tol_disp_ref_beta=ap.dqf,
        tol_energy=0,
        tol_beta_beating=ap.beta_beating,
        halo_x=n1, #ap.halo_h,
        halo_y=n1, #ap.halo_v,
        halo_r=n1, #ap.halo_r,
        halo_primary=ap.halo_prim,
    )
    twiss_data = clib.G2DTwissData(
        x=row.x,
        y=row.y,
        betx=row.betx,
        bety=row.bety,
        dx=row.dx,
        dy=row.dy,
        delta=ap.twiss_deltap,
        gamma=ap.gamma,
    )
    pts = path.get_points(ds_min=None)
    aper_data = clib.G2DBeamApertureData(
        points=pts,
        n_points=len(pts),
        tol_x=row.xtol,
        tol_y=row.ytol,
        tol_r=row.rtol,
    )
    halo_pts = clib.geom2d_get_beam_envelope(beam_data, twiss_data, aper_data, len_points=30)
    ax.plot(halo_pts['x'], halo_pts['y'], color='pink', label=fr'beam with halo = (x = n1$\sigma$, y = n1$\sigma$, r = n1$\sigma$)) [cgeom]')
    # Plot pos tol
    d_arc = ap.dqf * ap.paras_dx
    dx = ap.beta_beating * d_arc * np.sqrt(row.betx / ap.betaqfx) * ap.twiss_deltap
    dy = ap.beta_beating * d_arc * np.sqrt(row.bety / ap.betaqfx) * ap.twiss_deltap
    co = ap.co_radius
    rtolx = row.rtol
    rtoly = row.rtol
    beam_re = Path2D.from_racetrack(
        halfhside=row.xtol + rtolx,
        halfvside=row.ytol + rtoly,
        rx=rtoly,
        ry=rtoly,
    )

    tol_points = beam_re.get_points(ds_min=None)
    off_x = ap.twiss_deltap * row.dx
    off_y = ap.twiss_deltap * row.dy
    ax.plot(row.x + tol_points['x'] + off_x, row.y + tol_points['y'] + off_y, c='b', label='tol')
    ax.plot(row.x + tol_points['x'] - off_x, row.y + tol_points['y'] - off_y, c='b', label='tol')

    # Plot pos btol
    beam_re = Path2D.from_ellipse(
        rx=co + dx,
        ry=co + dy,
    )
    btol_points = beam_re.get_points(ds_min=None)
    ax.plot(row.x + btol_points['x'] + off_x, row.y + btol_points['y'] + off_y, c='y', label='btol')
    ax.plot(row.x + btol_points['x'] - off_x, row.y + btol_points['y'] - off_y, c='y', label='btol')


handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()
