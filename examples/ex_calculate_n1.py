import numpy as np
from matplotlib import pyplot as plt

from cgeom import Path2D, clib

# Beam envelope as a racetrack at 1 sigma at MQXFA.A1R1
# s = 19474.1156257836
# half_h_side = 0.005507120240906538
# half_v_side = 0.004944851543226073
# radius_x = 0.004507120240906538
# radius_y = 0.003944851543226073

n1_madx = 22.8284091641337
xtol = 0.001
ytol = 0.001
rtol = 0.0006
rxtol = 0.0026
rytol = 0.0026
sh = 0.0
sv = 0.0
sr = 1.0
sig_x = 0.0019071202409065373
sig_y = 0.001344851543226073
x = 1.90878474504131e-07
y = 0.0068072297363621


def beam_pts_sigma(n_sigma):
    scale = n_sigma
    radius_x = rxtol + sr * sig_x * scale
    half_h_side = (xtol + sh * sig_x * scale) + radius_x
    radius_y = rytol + sr * sig_y * scale
    half_v_side = (ytol + sv * sig_y * scale) + radius_y

    beam_path_1sig = Path2D.from_racetrack(
        halfhside=half_h_side,
        halfvside=half_v_side,
        rx=radius_x,
        ry=radius_y,
    )
    beam_pts = beam_path_1sig.get_points(ds_min=0.01)
    beam_pts['x'] += x
    beam_pts['y'] += y
    return beam_pts

# Aperture at MQXFA.A1R1
ap_half_width = 0.04747
ap_half_height = 0.04747
ap_diagonal = 0.04747
aperture_path = Path2D.from_octagon(ap_half_width, ap_half_height, ap_diagonal)
aper_poly = aperture_path.get_points(ds_min=None)

fig, ax = plt.subplots()
ax.plot(aper_poly['x'], aper_poly['y'], c='k')
beam_pts = beam_pts_sigma(1)
ax.plot(beam_pts['x'], beam_pts['y'], label=r'beam envelope at 1$\sigma$')

# Compute n1 in Python through bisection
def points_inside(t, c=None, **kwargs):
    beam_pts = beam_pts_sigma(t)
    inside = clib.geom2d_points_inside_polygon(points=beam_pts, poly_points=aper_poly)
    inside = int.from_bytes(inside)
    c = c or ('g' if inside else 'r')
    ax.plot(beam_pts['x'], beam_pts['y'], c=c, **kwargs)
    return inside

def bisect_inside(lo, hi, tol):
    while hi - lo > tol:
        mid = (hi + lo) / 2
        inside = points_inside(mid, linestyle='--')
        if inside:
            lo = mid
        else:
            hi = mid
    return lo

n1 = bisect_inside(15, 30, 0.001)

points_inside(n1, c='brown', label=rf'beam envelope at {n1:.3f}$\sigma$ [python]')

print(f'Computed n1 [python] is {n1}, difference vs MAD-X: abs = {np.abs(n1 - n1_madx):.4f}, rel = {np.abs(n1 - n1_madx) / n1_madx * 100:.2f}%')

# Compute n1 in C using the same method
beam_data = clib.G2DBeamData(
    emitx_norm=2.5e-06,
    emity_norm=2.5e-06,
    delta_rms=0,
    tol_co=0.002,
    tol_disp=0.1,
    tol_disp_ref_dx=0.1,
    tol_disp_ref_beta=2.086,
    tol_energy=0,
    tol_beta_beating=1.1,
    halo_x=6.001,
    halo_y=6,
    halo_r=6,
    halo_primary=6,
)
twiss_data = clib.G2DTwissData(
    x=x,
    y=y,
    betx = 8970.15631197269,
    bety = 4460.59251595666,
    dx = -0.0007896121555683,
    dy = 0.0295188586261303,
    delta=0,
    gamma=7460.52247352616,
)
aper_data = clib.G2DBeamApertureData(
    points=aper_poly,
    n_points=len(aper_poly),
    tol_x=xtol,
    tol_y=ytol,
    tol_r=rtol,
)
out_points, n1_c = clib.geom2d_compute_max_aperture_sigma_bisection(
    beam_data,
    twiss_data,
    aper_data,
    len_points=100,
    lower_bound=15,
    upper_bound=30,
    tol=0.01,
)

ax.plot(out_points['x'], out_points['y'], c='y', label=rf'beam envelope at {n1_c:.3f}$\sigma$ [cgeom]')
print(f'Computed n1 [cgeom] is {n1_c}.')

ax.set_aspect('equal')
plt.legend()
plt.show()
