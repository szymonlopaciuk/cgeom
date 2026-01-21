#ifndef CGEOM_BEAM_APERTURE_H
#define CGEOM_BEAM_APERTURE_H

#include "base.h"
#include "path.h"

typedef struct
{
    double x;     // closed orbit x
    double y;     // closed orbit y
    double betx;  // beta x
    double bety;  // beta y
    double dx;    // dispersion x
    double dy;    // dispersion y
    double delta; // relative energy deviation
    double gamma; // relativistic gamma
} G2DTwissData;

typedef struct {
    double emitx_norm;        // normalized emittance x
    double emity_norm;        // normalized emittance y
    double delta_rms;         // rms energy spread
    double tol_co;            // tolerance for closed orbit [co_radius]
    double tol_disp;          // tolerance for normalized dispersion [dqf]
    double tol_disp_ref_dx;   // tolerance for reference dispersion derivative [paras_dx]
    double tol_disp_ref_beta; // tolerance for reference dispersion beta [betaqfx]
    double tol_energy;        // tolerance for energy error [twiss_deltap]
    double tol_beta_beating;  // tolerance for beta beating in sigma [beta_beating]
    double halo_x;            // n sigma of horizontal halo
    double halo_y;            // n sigma of vertical halo
    double halo_r;            // n sigma of 45 degree halo
    double halo_primary;      // n sigma of primary halo
} G2DBeamData;

typedef struct {
    G2DPoint *points; // points defining the aperture shape
    int n_points;     // number of points defining the aperture shape
    double tol_r;     // radial tolerance for point-in-aperture check
    double tol_x;     // horizontal tolerance for point-in-aperture check
    double tol_y;     // vertical tolerance for point-in-aperture check
} G2DBeamApertureData;

void geom2d_get_beam_envelope(const G2DBeamData *beam_data, const G2DTwissData *twiss_data, const G2DBeamApertureData *aperture_data, int len_points, G2DPoint *out_points);
void generate_beam_envelope_from_sigma_xy(G2DBeamData *beam_data, G2DTwissData *twiss_data, G2DBeamApertureData *aperture_data, G2DSegment *segments, int n_segments);

char geom2d_is_point_inside_polygon(const G2DPoint* point, const G2DPoint* points, const int len_points);
char geom2d_points_inside_polygon(const G2DPoint* points, const G2DPoint* poly_points, const int len_points, const int len_poly_points);

double geom2d_compute_max_aperture_sigma_bisection(
    G2DBeamData *beam_data,
    const G2DTwissData *twiss_data,
    const G2DBeamApertureData *aperture_data,
    int len_points,
    const double lower_bound,
    const double upper_bound,
    const double tol,
    G2DPoint *out_points
);

#endif // CGEOM_BEAM_APERTURE_H
