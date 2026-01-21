#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "base.h"
#include "path.h"
#include "beam_aperture.h"

void geom2d_get_beam_envelope(const G2DBeamData *beam_data, const G2DTwissData *twiss_data, const G2DBeamApertureData *aperture_data, int len_points, G2DPoint *out_points)
/* Create beam envelope based on beam data, twiss data, aperture data is needed to get tolerances on the shape

See pyoptics/aperture.py: get_halo

Contract: len(out_points)=len_points
*/
{
    double x0 = twiss_data->x;  // assuming closed orbit relative to aperture center
    double y0 = twiss_data->y;  // assuming closed orbit relative to aperture center
    double betx = twiss_data->betx;
    double bety = twiss_data->bety;
    double dx = twiss_data->dx;
    double dy = twiss_data->dy;
    double gamma = twiss_data->gamma;

    double emitx_norm = beam_data->emitx_norm;
    double emity_norm = beam_data->emity_norm;
    double delta_rms = beam_data->delta_rms;
    double tol_beta_beating = beam_data->tol_beta_beating;
    double tol_disp_ref_beta = beam_data->tol_disp_ref_beta;
    double tol_disp = beam_data->tol_disp;
    double tol_disp_ref_dx = beam_data->tol_disp_ref_dx;
    double tol_co = beam_data->tol_co;

    double tol_r = aperture_data->tol_r;
    double tol_x = aperture_data->tol_x;
    double tol_y = aperture_data->tol_y;

    double hr = beam_data->halo_r;
    double hx = beam_data->halo_x;
    double hy = beam_data->halo_y;

    double ex = emitx_norm / gamma;
    double ey = emity_norm / gamma;

    double sigma_x = sqrt(ex * betx + dx * dx * delta_rms * delta_rms) * tol_beta_beating;
    double sigma_y = sqrt(ey * bety + dy * dy * delta_rms * delta_rms) * tol_beta_beating;

    double tol_dx = tol_beta_beating * tol_disp * tol_disp_ref_dx * (betx / tol_disp_ref_beta) * delta_rms;
    double tol_dy = tol_beta_beating * tol_disp * tol_disp_ref_dx * (betx / tol_disp_ref_beta) * delta_rms;

    double tol_rx = tol_r + tol_co + tol_dx;
    double tol_ry = tol_r + tol_co + tol_dy;

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
    double tmp = sqrt(2) * sqrt((hr - hx) * (hr - hy));
    double sh = hr - hy + tmp;
    double sv = hr - hx + tmp;
    double sr = hx + hy - hr - tmp;

    /*
        Remembering that hx, hy, and hr are specified in sigmas, we convolve the
        beam racetrack with the aperture tolerance racetrack, to get our beam
        envelope racetrack:
    */
    double h = tol_x + sh * sigma_x;
    double v = tol_y + sv * sigma_y;
    double a = tol_rx + sr * sigma_x;
    double b = tol_ry + sr * sigma_y;

    G2DSegment segments[8];
    G2DPath path;
    path.segments = segments;
    path.len_segments = 8;
    geom2d_segments_from_racetrack(h + a, v + b, a, b, path.segments, &path.len_segments);
    geom2d_path_get_n_uniform_points(&path, len_points, out_points);
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
    const double dx = b->x - a->x;
    const double dy = b->y - a->y;

    const double lhs = dx * (q->y - a->y);
    const double rhs = (q->x - a->x) * dy;

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


void set_halo_sigmas(G2DBeamData* beam_data, const double n)
{
    beam_data->halo_r = n;
    beam_data->halo_x = n;
    beam_data->halo_y = n;
}


double geom2d_compute_max_aperture_sigma_bisection(
    G2DBeamData *beam_data,
    const G2DTwissData *twiss_data,
    const G2DBeamApertureData *aperture_data,
    int len_points,
    const double lower_bound,
    const double upper_bound,
    const double tol,
    G2DPoint *out_points
)
/* Obtain the maximum number of sigmas that the beam fits within the aperture.

Contract: len(out_points)=len_points; len_poly_points=len(poly_points)
*/
{
    const double tmp_hr = beam_data->halo_r;
    const double tmp_hx = beam_data->halo_x;
    const double tmp_hy = beam_data->halo_y;

    const G2DPoint* poly_points = aperture_data->points;
    const double len_poly_points = aperture_data->n_points;

    double lo = lower_bound;
    double hi = upper_bound;

    while (hi - lo > tol) {
        const double mid = (lo + hi) / 2;
        set_halo_sigmas(beam_data, mid);
        geom2d_get_beam_envelope(beam_data, twiss_data, aperture_data, len_points, out_points);
        char inside = geom2d_points_inside_polygon(out_points, poly_points, len_points, len_poly_points);

        if (inside) lo = mid;
        else hi = mid;
    }

    beam_data->halo_r = tmp_hr;
    beam_data->halo_x = tmp_hx;
    beam_data->halo_y = tmp_hy;

    return lo;
}