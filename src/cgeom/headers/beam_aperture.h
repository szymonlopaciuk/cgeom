#ifndef CGEOM_BEAM_APERTURE
#define CGEOM_BEAM_APERTURE

#include <math.h>
#include <stdlib.h>

#include "xobjects/headers/common.h"
#include "base.h"
#include "path.h"


typedef struct
{
    float_type x;     // closed orbit x
    float_type y;     // closed orbit y
    float_type betx;  // beta x
    float_type bety;  // beta y
    float_type dx;    // dispersion x
    float_type dy;    // dispersion y
    float_type delta; // relative energy deviation
    float_type gamma; // relativistic gamma
} G2DTwissData;


typedef struct {
    float_type emitx_norm;        // normalized emittance x
    float_type emity_norm;        // normalized emittance y
    float_type delta_rms;         // rms energy spread
    float_type tol_co;            // tolerance for closed orbit [co_radius]
    float_type tol_disp;          // tolerance for normalized dispersion [dqf]
    float_type tol_disp_ref_dx;   // tolerance for reference dispersion derivative [paras_dx]
    float_type tol_disp_ref_beta; // tolerance for reference dispersion beta [betaqfx]
    float_type tol_energy;        // tolerance for energy error [twiss_deltap]
    float_type tol_beta_beating;  // tolerance for beta beating in sigma [beta_beating]
    float_type halo_x;            // n sigma of horizontal halo
    float_type halo_y;            // n sigma of vertical halo
    float_type halo_r;            // n sigma of 45 degree halo
    float_type halo_primary;      // n sigma of primary halo
} G2DBeamData;


typedef struct {
    G2DPoint *points; // points defining the aperture shape
    int n_points;     // number of points defining the aperture shape
    float_type tol_r;     // radial tolerance for point-in-aperture check
    float_type tol_x;     // horizontal tolerance for point-in-aperture check
    float_type tol_y;     // vertical tolerance for point-in-aperture check
} G2DBeamApertureData;


void geom2d_get_beam_envelope(
    const G2DBeamData *beam_data,
    const G2DTwissData *twiss_data,
    const G2DBeamApertureData *aperture_data,
    const float_type num_sigmas,
    int len_points,
    G2DPoint *out_points
)
/* Create beam envelope based on beam data, twiss data, aperture data is needed to get tolerances on the shape

See pyoptics/aperture.py: get_halo

Contract: len(out_points)=len_points
*/
{
    float_type x0 = twiss_data->x;  // assuming closed orbit relative to aperture center
    float_type y0 = twiss_data->y;  // assuming closed orbit relative to aperture center
    float_type betx = twiss_data->betx;
    float_type bety = twiss_data->bety;
    float_type dx = twiss_data->dx;
    float_type dy = twiss_data->dy;
    float_type gamma = twiss_data->gamma;

    float_type emitx_norm = beam_data->emitx_norm;
    float_type emity_norm = beam_data->emity_norm;
    float_type delta_rms = beam_data->delta_rms;
    float_type tol_beta_beating = beam_data->tol_beta_beating;
    float_type tol_disp_ref_beta = beam_data->tol_disp_ref_beta;
    float_type tol_disp = beam_data->tol_disp;
    float_type tol_disp_ref_dx = beam_data->tol_disp_ref_dx;
    float_type tol_co = beam_data->tol_co;

    float_type tol_r = aperture_data->tol_r;
    float_type tol_x = aperture_data->tol_x;
    float_type tol_y = aperture_data->tol_y;

    float_type hr = num_sigmas * beam_data->halo_r / beam_data->halo_primary;
    float_type hx = num_sigmas * beam_data->halo_x / beam_data->halo_primary;
    float_type hy = num_sigmas * beam_data->halo_y / beam_data->halo_primary;

    float_type ex = emitx_norm / gamma;
    float_type ey = emity_norm / gamma;

    float_type sigma_x = sqrt(ex * betx + dx * dx * delta_rms * delta_rms) * tol_beta_beating;
    float_type sigma_y = sqrt(ey * bety + dy * dy * delta_rms * delta_rms) * tol_beta_beating;

    float_type tol_dx = tol_beta_beating * tol_disp * tol_disp_ref_dx * sqrt(betx / tol_disp_ref_beta) * delta_rms;
    float_type tol_dy = tol_beta_beating * tol_disp * tol_disp_ref_dx * sqrt(bety / tol_disp_ref_beta) * delta_rms;

    float_type tol_rx = tol_r + tol_co + tol_dx;
    float_type tol_ry = tol_r + tol_co + tol_dy;

    /*
        We describe the beam of the shape described by hx, hy, and hr as a
        racetrack, leading to the following equations, where sh and sv are the
        width and height of a rectangle, which, when convolved with a circle of
        radius sr (Minkowski sum) yields our racetrack:

        { hx = sh + sr  (width),
        { hy = sv + sr  (height),
        { hr = sqrt(sh ** 2 + sv ** 2) + sr  (radial maximum: through (sh, sv)).

        Solving these for sh, sv, and sr yields the following equations:
    */
    float_type tmp = sqrt(2) * sqrt((hr - hx) * (hr - hy));
    float_type sh = hr - hy + tmp;
    float_type sv = hr - hx + tmp;
    float_type sr = hx + hy - hr - tmp;

    /*
        Remembering that hx, hy, and hr are specified in sigmas, we convolve the
        beam racetrack with the aperture tolerance racetrack, to get our beam
        envelope racetrack:
    */
    float_type h = tol_x + sh * sigma_x;
    float_type v = tol_y + sv * sigma_y;
    float_type a = tol_rx + sr * sigma_x;
    float_type b = tol_ry + sr * sigma_y;

    G2DSegment segments[8];
    G2DPath path;
    path.segments = segments;
    path.len_segments = 8;
    geom2d_segments_from_racetrack(h + a, v + b, a, b, path.segments, &path.len_segments);
    geom2d_poly_get_n_uniform_points(&path, len_points, out_points);
    geom2d_points_translate(x0, y0, out_points, len_points);
}


char horizontal_ray_intersects_segment(const G2DPoint* q, const G2DPoint* a, const G2DPoint* b)
{
    // Straddle test
    const int above_a = (a->y > q->y);
    const int above_b = (b->y > q->y);
    if (above_a == above_b) return 0;

    /* We are within the horizontal "strip" delimited by `a.y` and `b.y`.

       To check the intersection, we compare the tangent of ab segment and
       the aq segment (here assuming `b` above `a`, otherwise we need to flip
       the comparison -- done on the `return` line):

           tan_segment = (b.y - a.y) / (b.x - a.x)
           tan_point = (q.y - a.y) / (q.x - a.x)
           intersects = tan_point >= tan_segment

       To avoid division by zero we can cross-multiply:
    */
    const float_type dx = b->x - a->x;
    const float_type dy = b->y - a->y;

    const float_type lhs = dx * (q->y - a->y);
    const float_type rhs = (q->x - a->x) * dy;

    return (dy > 0) ? (lhs > rhs) : (lhs < rhs);
}


char geom2d_is_point_inside_polygon(const G2DPoint* point, const G2DPoint* points, const int len_points)
/* Determine if a point is inside a polygon.

Assume the polygon is closed, i.e. that points[-1] == points[0].

Contract: len_points=len(points)
*/
{
    char inside = 0;
    for (int i = 0; i < len_points - 1; i++)
    {
        const G2DPoint* a = &points[i];
        const G2DPoint* b = &points[i + 1];
        inside ^= horizontal_ray_intersects_segment(point, a, b);
    }

    // If count is odd, point is inside (return true), otherwise return false
    return inside;
}


char _is_point_inside_polygon(const float_type* point, const float_type* points, const int len_points)
{
    return geom2d_is_point_inside_polygon((const G2DPoint*) point, (const G2DPoint*) points, len_points);
}


char geom2d_points_inside_polygon(const G2DPoint* points, const G2DPoint* poly_points, const int len_points, const int len_poly_points)
/* Given a set of point, determine if they are inside a polygon. False if there
is at least one point outside of the polygon, and true if all points
are contained in the polygon.

Assume the polygon is closed, i.e. that poly_points[-1] == poly_points[0].

Contract: len_points=len(points); len_poly_points=len(poly_points)
*/
{
    for (int i = 0; i < len_points; i++)
    {
        const G2DPoint point = points[i];
        if (!geom2d_is_point_inside_polygon(&point, poly_points, len_poly_points))
            return 0;
    }
    return 1;
}


char _points_inside_polygon(const float_type* points, const float_type* poly_points, const int len_points, const int len_poly_points)
{
    return geom2d_points_inside_polygon((const G2DPoint*) points, (const G2DPoint*) poly_points, len_points, len_poly_points);
}


float_type geom2d_compute_max_aperture_sigma_bisection(
    G2DBeamData *beam_data,  // TODO: NOT THREAD SAFE if beam_data is shared!!!
    const G2DTwissData *twiss_data,
    const G2DBeamApertureData *aperture_data,
    int len_points,
    const float_type lower_bound,
    const float_type upper_bound,
    const float_type tol,
    G2DPoint *out_points
)
/* Obtain the maximum number of sigmas that the beam fits within the aperture.

Contract: len(out_points)=len_points; len_poly_points=len(poly_points)
*/
{
    const G2DPoint* poly_points = aperture_data->points;
    const float_type len_poly_points = aperture_data->n_points;

    float_type lo = lower_bound;
    float_type hi = upper_bound;

    while (hi - lo > tol) {
        const float_type mid = (lo + hi) / 2;
        geom2d_get_beam_envelope(beam_data, twiss_data, aperture_data, mid, len_points, out_points);
        char inside = geom2d_points_inside_polygon(out_points, poly_points, len_points, len_poly_points);

        if (inside) lo = mid;
        else hi = mid;
    }

    return lo;
}


uint32_t find_cross_section_for_s_bisection(
    const CrossSections cross_sections,
    const float_type target_s,
    float_type* const points
)
{
    const uint32_t num_cross_sections = CrossSections_get_count(cross_sections);

    uint32_t lo = 0;
    uint32_t hi = num_cross_sections;

    while (hi > lo) {
        const uint32_t mid = lo + (hi - lo) / 2;
        float_type current_s = CrossSections_get_s_positions(cross_sections, mid);

        if (current_s <= target_s) lo = mid + 1;
        else hi = mid;
    }

    const uint32_t found_idx = (lo == 0 ? 0 : lo - 1);
    return found_idx;
}


uint32_t find_cross_section_for_s_left_to_right(
    const CrossSections cross_sections,
    const float_type target_s,
    float_type* const points,
    const uint32_t lower_bound
)
{
    const uint32_t num_cross_sections = CrossSections_get_count(cross_sections);

    uint32_t found_idx = lower_bound;

    for (uint32_t i = lower_bound; i < num_cross_sections; ++i) {
        float_type current_s = CrossSections_get_s_positions(cross_sections, i);

        if (current_s <= target_s) {
            found_idx = i;
        } else {
            break;
        }
    }
    return found_idx;
}


void interpolate_profile(
    const CrossSections cross_sections,
    const uint32_t idx,
    float_type* const points,
    const float_type target_s
)
{
    // TODO: Implement proper interpolation: for now we simply copy the profile to the left
    const uint32_t num_points = CrossSections_get_num_points(cross_sections);
    float_type* const found_cross_section = CrossSections_getp3_points(cross_sections, idx, 0, 0);
    memcpy(points, found_cross_section, 2 * num_points * sizeof(float_type));

    if (points[0] != points[0]) {
        float_type s = CrossSections_get_s_positions(cross_sections, idx);
        fflush(stdout);
    }
}


uint32_t find_cross_section_for_s(
    const CrossSections cross_sections,
    const float_type target_s,
    float_type* const points,
    const uint32_t lower_bound
)
{
    uint32_t found_idx;
    if (lower_bound == 0) {
        // If starting from the left, let's do a bisection because chances are we need to jump to a random point
        found_idx = find_cross_section_for_s_bisection(cross_sections, target_s, points);
    }
    else {
        // If a lower bound is specified, chances are the next point that we search for is close to the right
        found_idx = find_cross_section_for_s_left_to_right(cross_sections, target_s, points, lower_bound);
    }

    interpolate_profile(cross_sections, found_idx, points, target_s);

    return found_idx;
}



void compute_max_aperture_sigma(
    ApertureModel model,
    CrossSections cross_sections,
    TwissData twiss_data,
    BeamData beam_data,
    float_type* const out_interpolated_apertures,
    const uint32_t envelope_num_points,
    float_type* const out_envelope_at_max_sigma,
    float_type* const sigmas
) {
    const uint32_t num_slices = TwissData_len_x(twiss_data);
    const uint32_t num_points = CrossSections_get_num_points(cross_sections);

    G2DBeamData s_beam_data = {
        .emitx_norm = BeamData_get_emitx_norm(beam_data),
        .emity_norm = BeamData_get_emity_norm(beam_data),
        .delta_rms = BeamData_get_delta_rms(beam_data),
        .tol_co = BeamData_get_tol_co(beam_data),
        .tol_disp = BeamData_get_tol_disp(beam_data),
        .tol_disp_ref_dx = BeamData_get_tol_disp_ref_dx(beam_data),
        .tol_disp_ref_beta = BeamData_get_tol_disp_ref_beta(beam_data),
        .tol_energy = BeamData_get_tol_energy(beam_data),
        .tol_beta_beating = BeamData_get_tol_beta_beating(beam_data),
        .halo_x = BeamData_get_halo_x(beam_data),
        .halo_y = BeamData_get_halo_y(beam_data),
        .halo_r = BeamData_get_halo_r(beam_data),
        .halo_primary = BeamData_get_halo_primary(beam_data)
    };


    #ifdef XO_CONTEXT_CPU
        int completed = 0;
    #endif

    VECTORIZE_OVER(idx_slice, num_slices)
    {
        uint32_t cross_section_index = 0;
        float_type* const points = out_interpolated_apertures + idx_slice * num_points * 2;
        float_type s = TwissData_get_s(twiss_data, idx_slice);

        const G2DTwissData s_twiss_data = {
            .x = TwissData_get_x(twiss_data, idx_slice),
            .y = TwissData_get_y(twiss_data, idx_slice),
            .betx = TwissData_get_betx(twiss_data, idx_slice),
            .bety = TwissData_get_bety(twiss_data, idx_slice),
            .dx = TwissData_get_dx(twiss_data, idx_slice),
            .dy = TwissData_get_dy(twiss_data, idx_slice),
            .delta = TwissData_get_delta(twiss_data, idx_slice),
            .gamma = TwissData_get_gamma(twiss_data)
        };

        cross_section_index = find_cross_section_for_s(cross_sections, s, points, cross_section_index);

        const uint32_t type_pos_idx = CrossSections_get_type_position_indices(cross_sections, cross_section_index);
        const uint32_t profile_pos_idx = CrossSections_get_profile_position_indices(cross_sections, cross_section_index);
        const uint32_t profile_idx = ApertureModel_get_types_positions_profile_index(model, type_pos_idx, profile_pos_idx);
        const Profile profile = ApertureModel_getp1_profiles(model, profile_idx);
        const float_type tol_r = Profile_get_tol_r(profile);
        const float_type tol_x = Profile_get_tol_x(profile);
        const float_type tol_y = Profile_get_tol_y(profile);

        const G2DBeamApertureData s_aperture_data = {
            .points = (G2DPoint* const)points,
            .n_points = num_points,
            .tol_r = tol_r,
            .tol_x = tol_x,
            .tol_y = tol_y
        };

        const float_type num_sigmas = geom2d_compute_max_aperture_sigma_bisection(
            &s_beam_data,
            &s_twiss_data,
            &s_aperture_data,
            envelope_num_points,
            /* lower bound on search */ 0,
            /* upper bound on search */ 10000,
            /* tolerance on search */ 0.01,
            (G2DPoint*)(out_envelope_at_max_sigma + idx_slice * envelope_num_points * 2)
        );
        sigmas[idx_slice] = num_sigmas;

        #ifdef XO_CONTEXT_CPU
            printf("Computing sigmas: %d%%\r", 100 * (++completed) / num_slices);
            fflush(stdout);
        #endif
    }
    END_VECTORIZE;
}

#endif /* CGEOM_BEAM_APERTURE */
