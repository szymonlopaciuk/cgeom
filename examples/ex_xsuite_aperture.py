import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt

from cgeom.aperture import Aperture

context = xo.ContextCpu(omp_num_threads='auto')

lhc_with_metadata = xt.load('./lhc/lhc_aperture.json')
b1 = lhc_with_metadata['b1']
lhc_length = b1.get_length()

aperture_model = Aperture.from_line_with_madx_metadata(b1, line_name='b1', context=context)

mqxfa_name = 'mqxfa.a1r1/b1'
# sigmas_mqxfa, tw, interpolated_points, envelope_at_max_sigma = aperture_model.get_aperture_sigmas_at_element(
#     line_name="b1",
#     element_name=mqxfa_name,
#     resolution=0.1,
#     cross_sections_num_points=100,
#     envelopes_num_points=36,
#     method='bisection',
# )

# sigmas_bisect, tw, interpolated_points, envelope_at_max_sigma = aperture_model.get_aperture_sigmas_at_s(
#     line_name="b1",
#     s_positions=np.linspace(0, lhc_length, 1000),
#     cross_sections_num_points=36,
#     envelopes_num_points=36,
#     method='bisection',
# )

sigmas_rays, tw, interpolated_points, envelope_at_max_sigma = aperture_model.get_aperture_sigmas_at_s(
    line_name="b1",
    s_positions=np.linspace(0, lhc_length, 1000),
    cross_sections_num_points=36,
    envelopes_num_points=36,
    method='rays',
)

# plt.plot(tw.s, sigmas_rays[:, 0], label='$n_h$')
# plt.plot(tw.s, sigmas_rays[:, 1], label='$n_v$')
# plt.plot(tw.s, sigmas_rays[:, 2], label='$n_d$')
plt.plot(tw.s, np.min(sigmas_rays, axis=1), label='$~n_1$')

# plt.plot(tw.s, sigmas_bisect, label='$n_1$')

# plt.plot(tw.s, np.max(np.abs(interpolated_points[:, :, 0]), axis=1), label='max x aperture')
# plt.plot(tw.s, np.max(np.abs(interpolated_points[:, :, 1]), axis=1), label='max y aperture')
plt.legend()
plt.show()

# plt.scatter(tw.x, tw.y)
#
# for pt in interpolated_points:
#     plt.plot(pt[:, 0], pt[:, 1], c='k')
#
# for pt in envelope_at_max_sigma:
#     plt.plot(pt[:, 0], pt[:, 1])
#
# plt.gca().set_aspect('equal')
# plt.legend()
# plt.show()

# for the MBRC lofting is needed
# look into ex_interpolate.py
