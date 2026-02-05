#include <stdlib.h>
#include <math.h>

#include "base.h"
#include "path.h"


void build_polygon_for_profile(const CrossSections, const uint64_t, const Profile);
void polygon_transform_in_type_frame(const CrossSections, const uint64_t, const ProfilePosition);
void build_circle_polygon(const CrossSections, const uint64_t, const Circle);
void build_rect_ellipse_polygon(const CrossSections, const uint64_t, const RectEllipse);
void build_racetrack_polygon(const CrossSections, const uint64_t, const Racetrack);
void build_octagon_polygon(const CrossSections, const uint64_t, const Octagon);


void build_profile_polygons(const ApertureModel model, const CrossSections cross_sections)  // TODO: include survey related logic
{
    /*
        Based on the aperture model and cross section location data, generate correct polygons, orthogonal to the
        survey s path.
    */
    const uint32_t num_cross_sections = CrossSections_get_count(cross_sections);

    for (uint32_t idx = 0; idx < num_cross_sections; idx++)
    {
        const uint32_t type_pos_idx = CrossSections_get_type_position_indices(cross_sections, idx);
        const uint32_t profile_pos_idx = CrossSections_get_profile_position_indices(cross_sections, idx);

        const TypePosition type_pos = ApertureModel_getp1_type_positions(model, type_pos_idx);
        const uint32_t type_idx = TypePosition_get_type_index(type_pos);
        const ApertureType aper_type = ApertureModel_getp1_types(model, type_idx);

        const ProfilePosition profile_pos = ApertureType_getp1_positions(aper_type, profile_pos_idx);
        const uint32_t profile_idx = ProfilePosition_get_profile_index(profile_pos);
        const Profile profile = ApertureModel_getp1_profiles(model, profile_idx);

        build_polygon_for_profile(cross_sections, idx, profile);
        polygon_transform_in_type_frame(cross_sections, idx, profile_pos);
    }
}


void build_polygon_for_profile(
    const CrossSections cross_sections,
    const uint64_t cross_section_idx,
    const Profile profile
) {
    /*
        Convert the logical description of a profile to a polygon, and store it in ``cross_sections``.
    */
    const uint64_t profile_type_id = Profile_typeid(profile);

    switch (profile_type_id)
    {
        case Profile_Circle_t:  // LHC
        {
            const Circle circle = Profile_member(profile);
            build_circle_polygon(cross_sections, cross_section_idx, circle);
            break;
        }
        case Profile_Rectangle_t:
        {
            const Rectangle rectangle = Profile_member(profile);
            build_rectangle_polygon(cross_sections, cross_section_idx, rectangle);
            break;
        }
        case Profile_Ellipse_t:
        {
            const Ellipse ellipse = Profile_member(profile);
            // TODO: Implement
            break;
        }
        case Profile_RectEllipse_t:
        {
            const RectEllipse rect_ellipse = Profile_member(profile);
            build_rect_ellipse_polygon(cross_sections, cross_section_idx, rect_ellipse);
            break;
        }
        case Profile_Racetrack_t:
        {
            const Racetrack racetrack = Profile_member(profile);
            build_racetrack_polygon(cross_sections, cross_section_idx, racetrack);
            break;
        }
        case Profile_Octagon_t:
        {
            const Octagon octagon = Profile_member(profile);
            build_octagon_polygon(cross_sections, cross_section_idx, octagon);
            break;
        }
        case Profile_Polygon_t:
        {
            const Polygon polygon = Profile_member(profile);
            // TODO: Implement
            break;
        }
        case Profile_SVGShape_t:
        {
            const SVGShape svg_shape = Profile_member(profile);
            // TODO: Implement
            break;
        }
    }
}


void polygon_transform_in_type_frame(
    const CrossSections cross_sections,
    const uint64_t idx,
    const ProfilePosition profile_pos
) {
    /*
        Apply the type frame transformation described in ``profile_pos`` to a polygon.
    */
    G2DPoint* const points = (G2DPoint* const) CrossSections_getp3_points(cross_sections, idx, 0, 0);
    const uint32_t num_points = CrossSections_get_num_points(cross_sections);

    const float shift_x = ProfilePosition_get_shift_x(profile_pos);
    const float shift_y = ProfilePosition_get_shift_y(profile_pos);

    for (uint32_t i; i < num_points; i++)
    {
        points[i].x += shift_x;
        points[i].y += shift_y;
        // TODO: Apply rotations, will change s (a heuristic for when a profile generates 1 or 2 cross sections needed?)
        // TODO: Also, how will we select if this is the entry or exit in case of 2 cross sections?
    }
}


void build_circle_polygon(const CrossSections cross_sections, const uint64_t idx, const Circle circle)
{
    G2DPoint* const points = (G2DPoint* const) CrossSections_getp3_points(cross_sections, idx, 0, 0);
    const uint32_t num_points = CrossSections_get_num_points(cross_sections);

    const float radius = Circle_get_radius(circle);

    G2DSegment segments[1];
    G2DPath path = {
        .segments = segments,
        .len_segments = 1
    };
    geom2d_segments_from_circle(radius, segments);
    geom2d_path_get_n_uniform_points(&path, num_points, points);
}


void build_rectangle_polygon(const CrossSections cross_sections, const uint64_t idx, const Rectangle rectangle)
{
    G2DPoint* const points = (G2DPoint* const) CrossSections_getp3_points(cross_sections, idx, 0, 0);
    const uint32_t num_points = CrossSections_get_num_points(cross_sections);

    const float half_width = Rectangle_get_half_width(rectangle);
    const float half_height = Rectangle_get_half_height(rectangle);

    G2DSegment segments[4];
    G2DPath path = {
        .segments = segments,
        .len_segments = 4
    };
    geom2d_segments_from_rectangle(half_width, half_height, segments);
    geom2d_path_get_n_uniform_points(&path, num_points, points);
}


void build_rect_ellipse_polygon(const CrossSections cross_sections, const uint64_t idx, const RectEllipse rect_ellipse)
{
    G2DPoint* const points = (G2DPoint* const) CrossSections_getp3_points(cross_sections, idx, 0, 0);
    const uint32_t num_points = CrossSections_get_num_points(cross_sections);

    const float half_width = RectEllipse_get_half_width(rect_ellipse);
    const float half_height = RectEllipse_get_half_height(rect_ellipse);
    const float half_major = RectEllipse_get_half_major(rect_ellipse);
    const float half_minor = RectEllipse_get_half_minor(rect_ellipse);

    G2DSegment segments[8];
    G2DPath path = {
        .segments = segments,
        .len_segments = 8
    };
    geom2d_segments_from_rectellipse(half_width, half_height, half_major, half_minor, segments, &path.len_segments);
    geom2d_path_get_n_uniform_points(&path, num_points, points);
}


void build_racetrack_polygon(const CrossSections cross_sections, const uint64_t idx, const Racetrack racetrack)
{
    G2DPoint* const points = (G2DPoint* const) CrossSections_getp3_points(cross_sections, idx, 0, 0);
    const uint32_t num_points = CrossSections_get_num_points(cross_sections);

    const float half_width = Racetrack_get_half_width(racetrack);
    const float half_height = Racetrack_get_half_height(racetrack);
    const float half_major = Racetrack_get_half_major(racetrack);
    const float half_minor = Racetrack_get_half_minor(racetrack);

    G2DSegment segments[8];
    G2DPath path = {
        .segments = segments,
        .len_segments = 8
    };
    geom2d_segments_from_racetrack(half_width, half_height, half_major, half_minor, segments, &path.len_segments);
    geom2d_path_get_n_uniform_points(&path, num_points, points);
}

void build_octagon_polygon(const CrossSections cross_sections, const uint64_t idx, const Octagon octagon)
{
    G2DPoint* const points = (G2DPoint* const) CrossSections_getp3_points(cross_sections, idx, 0, 0);
    const uint32_t num_points = CrossSections_get_num_points(cross_sections);

    const float half_width = Octagon_get_half_width(octagon);
    const float half_height = Octagon_get_half_height(octagon);
    const float half_diagonal = Octagon_get_half_diagonal(octagon);

    G2DSegment segments[8];
    G2DPath path = {
        .segments = segments,
        .len_segments = 8
    };
    geom2d_segments_from_octagon(half_width, half_height, half_diagonal, segments, &path.len_segments);
    geom2d_path_get_n_uniform_points(&path, num_points, points);
}
