import bisect
import re
from collections import defaultdict
from functools import singledispatchmethod, lru_cache
from typing import Dict, Union, Tuple, List, Optional, Iterator, Any

import numpy as np

import xobjects as xo
from xtrack.beam_elements import apertures
from xtrack.progress_indicator import progress
from xtrack.environment import Environment


LimitTypes = Union[
    apertures.LimitRect,
    apertures.LimitEllipse,
    apertures.LimitRectEllipse,
    apertures.LimitRacetrack,
    apertures.LimitPolygon,
]


class Circle(xo.Struct):
    radius = xo.Float32


class Rectangle(xo.Struct):
    half_width = xo.Float32
    half_height = xo.Float32


class Ellipse(xo.Struct):
    half_major = xo.Float32
    half_minor = xo.Float32


class RectEllipse(xo.Struct):
    max_x = xo.Float32
    max_y = xo.Float32
    half_major = xo.Float32
    half_minor = xo.Float32


class Racetrack(xo.Struct):
    half_width = xo.Float32
    half_height = xo.Float32
    half_major = xo.Float32
    half_minor = xo.Float32


class Octagon(xo.Struct):
    half_width = xo.Float32
    half_height = xo.Float32
    half_diagonal = xo.Float32


class Polygon(xo.Struct):
    vertices = xo.Float32[:, 2]


class SVGShape(xo.Struct):
    svg_data = xo.String


ProfileTypes = Union[Circle, Rectangle, Ellipse, RectEllipse, Racetrack, Octagon, Polygon, SVGShape]


class Profile(xo.UnionRef):
    _reftypes = [Circle, Rectangle, Ellipse, RectEllipse, Racetrack, Octagon, Polygon, SVGShape]


class ProfileList(xo.Struct):
    profiles = Profile[:]

    def __len__(self) -> int:
        return len(self.profiles)

    def __getitem__(self, idx: int) -> ProfileTypes:
        return self.profiles[idx]


class Profiles:
    """Associate profiles to names.

    Parameters
    ----------
    indices: dict
        Dictionary mapping names of profiles to indices in ``profiles``.
    profiles: ProfileList
        List of profiles.
    """
    def __init__(self, indices: Dict[str, int], profiles: ProfileList):
        if list(indices.values()) != list(range(len(profiles))):
            raise ValueError('Indices are expected to be ordered by value, and there should be one for each type.')

        self.indices = indices  # dict of profile names to indices
        self.profile_list = profiles  # list of profile objects

    def __getitem__(self, name: str) -> ProfileTypes:
        try:
            index = self.indices[name]
            return self.profile_list.profiles[index]
        except KeyError:
            raise KeyError(f"Profile {name} not found.")

    def name_for_index(self, idx: int) -> str:
        return list(self.indices.keys())[idx]


class ProfilePosition(xo.Struct):
    """Description of the placement of a profile in type (lab) frame.

    Parameters
    ----------
    profile_index: int
        The index identifying the profile in the associated ``Profiles`` object.
    s_position: float
        The position along the type axis where this profile sits.
    shift_x: float
        The horizontal shift of the profile center from the type axis.
    shift_y: float
        The vertical shift of the profile center from the type axis
    rot_x: float
        The rotation of the profile around the type axis in radians.
    rot_y: float
        The rotation of the profile around the vertical axis in radians.
    rot_z: float
        The rotation of the profile around the horizontal axis in radians.
    """
    profile_index = xo.Int64
    type_index = xo.Int64
    s_position = xo.Float32
    shift_x = xo.Float32
    shift_y = xo.Float32
    rot_x = xo.Float32
    rot_y = xo.Float32
    rot_z = xo.Float32

    def copy(self):
        return ProfilePosition(
            profile_index=self.profile_index,
            s_position=self.s_position,
            shift_x=self.shift_x,
            shift_y=self.shift_y,
            rot_x=self.rot_x,
            rot_y=self.rot_y,
            rot_z=self.rot_z,
        )


class ApertureType(xo.Struct):
    """Description of the type, i.e. a section consisting of pipes (profiles).

    Parameters
    ----------
    curvature: float
        curvature of the type axis assumed to be in the horizontal plane

    positions: List[ProfilePosition]
        The list of profile positions comprising the type.
    """
    curvature = xo.Float32
    positions = ProfilePosition[:]


class ApertureTypeList(xo.Struct):
    types = ApertureType[:]

    def __getitem__(self, idx: int) -> ApertureType:
        return self.types[idx]

    def __len__(self) -> int:
        return len(self.types)


class ApertureTypes:
    """Associate types to names.

    Parameters
    ----------
    indices: dict
        Dictionary mapping names of types to indices in ``types``. Should be ordered by value,
        and there should be one for each type.
    types: ApertureTypeList
        List of pipes in the lab frame.
    """
    def __init__(self, indices: Dict[str, int], types: ApertureTypeList):
        if list(indices.values()) != list(range(len(types))):
            raise ValueError('Indices are expected to be ordered by value, and there should be one for each type.')

        self.indices = indices
        self.types = types

    def name_for_index(self, idx: int) -> str:
        return list(self.indices.keys())[idx]


class TypePosition(xo.Struct):
    type_index = xo.Int32
    ref_position = xo.String  # identify a point in survey
    idx_position = xo.Int32  # index of the point in the survey
    transformation = xo.Float32[
        4, 4
    ]  # 3D rigid transformation matrix from the ref point to the center of the aperture type


class TypePositionList(xo.Struct):
    positions = TypePosition[:]


class ApertureModel(xo.Struct):
    def __init__(self, line_name: str, type_positions: TypePositionList):
        self.line_name = line_name
        self.type_positions = type_positions  # positioning of types in line frame


class PolygonalProfiles(xo.Float32[:, :, 2]):
    pass


def transform_matrix(dx, dy, ds, theta, phi, psi):
    """Generate a 3D transformation matrix.

    Parameters
    ----------
    dx, dy, ds : float
        Shifts in x, y, and s directions
    theta : float
        Rotation around the y-axis (positive s to x) in radians
    phi
        Rotation around the x-axis (positive s to y) in radians
    psi
        Rotation around the s-axis (positive y to x) in radians
    """
    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_theta, c_theta = np.sin(theta), np.cos(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)
    matrix = np.array(
        [
            [-s_phi * s_psi * s_theta + c_psi * c_theta,
                -c_psi * s_phi * s_theta - c_theta * s_psi, c_phi * s_theta, dx],
            [c_phi * s_psi, c_phi * c_psi, s_phi, dy],
            [-c_theta * s_phi * s_psi - c_psi * s_theta,
                -c_psi * c_theta * s_phi + s_psi * s_theta, c_phi * c_theta, ds],
            [0, 0, 0, 1],
        ]
    )
    return matrix


class Aperture:
    halo_params = {
        "emitx_norm": 3.5e-6,  # normalized emittance x
        "emity_norm": 3.5e-6,  # normalized emittance y
        "delta_rms": 0.0,  # rms energy spread
        "tol_co": 0.0,  # tolerance for closed orbit
        "tol_disp": 0.0,  # tolerance for normalized dispersion
        "tol_disp_ref_dx": 1.8,  # tolerance for reference dispersion derivative
        "tol_disp_ref_beta": 170,  # tolerance for reference dispersion beta
        "tol_energy": 0.0,  # tolerance for energy error
        "tol_beta_beating": 1.0,  # tolerance for beta beating in sigma
        "halo_x": 6.0,  # n sigma of horizontal halo
        "halo_y": 6.0,  # n sigma of vertical halo
        "halo_r": 6.0,  # n sigma of 45 degree halo
        "halo_primary": 6.0,  # n sigma of primary halo
    }

    def __init__(
        self,
        env,
        profiles: Profiles,
        aperture_types: ApertureTypes,
        aperture_model: ApertureModel,
        cross_sections,
        halo_params=None,
        s_tol=1e-3,
    ):
        self.env = env
        self.profiles = profiles  # list of profile objects
        self.aperture_types = aperture_types  # list of pipes in the lab frame
        self.aperture_model = aperture_model  # positioning of types in line frame
        self.cross_sections = cross_sections
        self.halo_params = self.halo_params.copy()
        self.s_tol = s_tol

        if halo_params is not None:
            self.halo_params.update(halo_params)

    @classmethod
    def from_line_with_madx_metadata(cls, line, line_name=None):
        env = line.env
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not an element
        name_to_sv_index = dict(zip(survey.name, range(len(survey))))
        layout_data = line.metadata['layout_data']

        name_iter_with_progress = progress(
            survey_names,
            desc="Building apertures",
            total=len(survey_names),
        )

        profiles = []
        types = []
        aperture_indices = {}
        type_positions_list = []

        for element_name in name_iter_with_progress:
            element = line.element_dict[element_name]

            # Discard line name suffix to get the aperture name
            aper_name = cls._guess_original_mad_name(element_name)
            if aper_name not in layout_data:
                continue

            element_metadata = layout_data[aper_name]

            if 'aperture' not in element_metadata:
                continue

            if aper_name not in aperture_indices:
                shape, params, tols = element_metadata['aperture']
                profile = cls._profile_from_madx_aperture(shape, params)

                if not profile:
                    # There is not really an aperture here, continue
                    continue

                assert len(types) == len(profiles)  # in MAD-X we will have just one type per profile

                aper_idx = len(types)
                aperture_indices[aper_name] = aper_idx

                profile_position = ProfilePosition(profile_index=aper_idx)
                offset_x, offset_y = element_metadata.get('offset', (0.0, 0.0))
                profile_position.shift_x = offset_x
                profile_position.shift_y = offset_y
                # TODO: any other transformations from metadata?

                if element.isthick:
                    # Place two profiles on either side of the element
                    profile_position_start = profile_position
                    profile_position_end = profile_position.copy()
                    profile_position_start.s_position = 0
                    profile_position_end.s_position = element.length
                    positions = [profile_position_start, profile_position_end]
                    curvature = getattr(element, 'h', 0)
                else:
                    # Place single profile at center of element
                    positions = [profile_position]
                    curvature = 0

                aperture_type = ApertureType(curvature=curvature, positions=positions)
                types.append(aperture_type)

                profiles.append(profile)

            # Apply element transformations to type position
            if element.transformations_active:
                matrix = transform_matrix(
                    dx=element.shift_x,
                    dy=element.shift_y,
                    ds=element.shift_s,
                    theta=element.rot_y_rad,
                    phi=element.rot_x_rad,
                    psi=element.rot_s_rad_no_frame,
                )
            else:
                matrix = np.identity(4)

            type_position = TypePosition(
                type_index=aperture_indices[aper_name],
                ref_position=element_name,
                idx_position=name_to_sv_index[element_name],
                transformation=matrix,
            )
            type_positions_list.append(type_position)

        aperture = cls._build_aperture_xobjects(
            env=env,
            line_name=line_name or line.name,
            type_indices=aperture_indices,
            type_list=types,
            type_position_list=type_positions_list,
            profile_indices=aperture_indices,
            profile_list=profiles,
        )
        return aperture

    @classmethod
    def from_line_with_associated_apertures(cls, line, line_name=None):
        env = line.env
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not an element
        name_to_sv_index = dict(zip(survey.name, range(len(survey_names))))

        profiles = []
        types = []
        aperture_indices = {}
        type_positions_list = []

        for survey_name in progress(survey_names, desc="Building aperture data", total=len(survey_names)):
            # Discard line name suffix to get the aperture name
            element = line[survey_name]
            aper_name = getattr(element, 'name_associated_aperture', None)

            if not aper_name:
                continue

            aper_element = line.element_dict[aper_name]

            if aper_name not in aperture_indices:
                profile, offset_x, offset_y = cls._profile_from_limit_element(aper_element)

                assert len(types) == len(profiles)  # in Xsuite with associated apertures we will have just one type per profile

                aper_idx = len(types)
                aperture_indices[aper_name] = aper_idx

                profile_position = ProfilePosition(profile_index=aper_idx)
                profile_position.shift_x = offset_x
                profile_position.shift_y = offset_y
                # TODO: any other transformations from metadata?

                if element.isthick:
                    # Place two profiles on either side of the element
                    profile_position_start = profile_position
                    profile_position_end = profile_position.copy()
                    profile_position_start.s_position = 0
                    profile_position_end.s_position = element.length
                    positions = [profile_position_start, profile_position_end]
                    curvature = getattr(element, 'h', 0)
                else:
                    # Place single profile at center of element
                    positions = [profile_position]
                    curvature = 0

                aperture_type = ApertureType(curvature=curvature, positions=positions)
                types.append(aperture_type)

                profiles.append(profile)

            # Apply element transformations to type position
            if element.transformations_active:
                # TODO: Need to correctly handle the situation where both the element and the aperture are misaligned.
                #  The matrix then needs to combine the two in a correct way. Curvature will probably complicate this
                #  even more.
                raise NotImplementedError('Aperture model not yet supported with element transformations.')

            if aper_element.transformations_active:
                matrix = transform_matrix(
                    dx=aper_element.shift_x,
                    dy=aper_element.shift_y,
                    ds=aper_element.shift_s,
                    theta=aper_element.rot_y_rad,
                    phi=aper_element.rot_x_rad,
                    psi=aper_element.rot_s_rad_no_frame,
                )
            else:
                matrix = np.identity(4)

            type_position = TypePosition(
                type_index=aperture_indices[aper_name],
                ref_position=survey_name,
                idx_position=name_to_sv_index[survey_name],
                transformation=matrix,
            )
            type_positions_list.append(type_position)

        aperture = cls._build_aperture_xobjects(
            env=env,
            line_name=line_name or line.name,
            type_indices=aperture_indices,
            type_list=types,
            type_position_list=type_positions_list,
            profile_indices=aperture_indices,
            profile_list=profiles,
        )
        return aperture

    @classmethod
    def from_line_with_limits(cls, line, line_name=None):
        env = line.env
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not a limit
        name_to_sv_index = dict(zip(survey.name, range(len(survey_names))))

        profiles = []
        type_list = []
        indices = {}
        type_positions_list = []

        aper_idx = 0

        for name in progress(survey_names, desc="Building aperture data", total=len(survey_names)):
            element = line[name]
            if not isinstance(element, LimitTypes):
                continue

            indices[name] = aper_idx
            profile, center_x, center_y = cls._profile_from_limit_element(element)
            profiles.append(profile)

            profile_position = ProfilePosition(profile_index=aper_idx)
            if element.transformations_active:
                profile_position.s_position = element.shift_s
                profile_position.shift_x = element.shift_x
                profile_position.shift_y = element.shift_y
                # TODO: Is this really how it should be??
                profile_position.rot_x = element.rot_s_rad_no_frame
                profile_position.rot_y = element.rot_x_rad
                profile_position.rot_z = element.rot_y_rad

            aperture_type = ApertureType(curvature=0, positions=[profile_position])
            type_list.append(aperture_type)

            type_position = TypePosition(
                type_index=aper_idx,
                ref_position=name,
                idx_position=name_to_sv_index[name],
                transformation=np.identity(4),
            )
            type_positions_list.append(type_position)

            aper_idx += 1

        aperture = cls._build_aperture_xobjects(
            env=env,
            line_name=line_name or line.name,
            type_indices=indices,
            type_list=type_list,
            type_position_list=type_positions_list,
            profile_indices=indices,
            profile_list=profiles,
        )
        return aperture

    @classmethod
    def _build_aperture_xobjects(
            cls,
            env: Environment,
            line_name: str,
            type_indices: Dict[str, int],
            type_list: List[ApertureType],
            type_position_list: List[TypePosition],
            profile_indices: Dict[str, int],
            profile_list: List[ProfileTypes],
    ) -> Aperture:
        """Build the Aperture class and its comprising xobjects.

        Parameters
        ----------
        env
            The environment of the line for which the aperture model is built.
        line_name
            The name of the line for which the aperture model is built.
        type_indices
            A mapping between the name of an aperture type and its index in ``type_list``.
        type_list
            List of aperture types featured in the model.
        type_position_list
            List of aperture type positions that define the model.
        profile_indices
            A mapping between the name of an aperture type and its index in ``profile_list``.
        profile_list
            List of all profiles featured in the model. The order must be consistent with the indices used inside
            each of the type definitions in ``type_list``.
        """
        common_buffer = xo.context_default.new_buffer()

        profile_list_xo = ProfileList(profiles=profile_list, _buffer=common_buffer)
        profiles = Profiles(indices=profile_indices, profiles=profile_list_xo)

        type_positions = TypePositionList(positions=type_position_list, _buffer=common_buffer)

        model = ApertureModel(
            line_name=line_name,
            type_positions=type_positions,
        )

        types = ApertureTypes(
            indices=type_indices,
            types=ApertureTypeList(types=type_list, _buffer=common_buffer),
        )

        aperture = cls(
            env=env,
            profiles=profiles,
            aperture_types=types,
            aperture_model=model,
            cross_sections=None,
        )

        return aperture

    def type_for_position(self, type_position: TypePosition) -> ApertureType:
        return self.aperture_types.types[type_position.type_index]

    def type_name_for_position(self, type_position: TypePosition) -> str:
        return self.aperture_types.name_for_index(type_position.type_index)

    def profile_for_position(self, profile_position: ProfilePosition) -> Profile:
        return self.profiles.profile_list.profiles[profile_position.profile_index]

    def profile_name_for_position(self, profile_position: ProfilePosition) -> str:
        return self.profiles.name_for_index(profile_position.profile_index)

    def get_aperture_sigma(self, line_name: str, element_name: str, resolution: float) -> np.ndarray:
        line = self.env[line_name]
        element = line[element_name]
        s_start = line.get_s_position(element_name)
        element_length = getattr(element, 'length', 0)
        s_end = s_start + element_length

        cuts = np.linspace(s_start, s_end, element_length / resolution)
        line_sliced = line.copy().cut_at_s(cuts)
        sliced_twiss = line_sliced.twiss()

        apertures = self._find_type_positions(s_start, s_end, line_name)

    def _polygons_at_s(self, s_positions: List[float], type_positions: List[TypePosition], num_points=50) -> PolygonalProfiles:
        polygons = PolygonalProfiles(len(s_positions), num_points)

        for idx, s in enumerate(s_positions):
            polygon_at_s = self._interpolate_polygon_at_s(s, type_positions, num_points)
            polygons[idx, :, :] = polygon_at_s

        return polygons

    def _interpolate_polygon_at_s(self, s_position: float, search_in: List[TypePosition], num_points: int) -> np.ndarray:
        # TODO: For now this just grabs the profile to the left of the current position. To do this properly,
        #  we need to handle interpolations between the profile to the left and to the right, being careful to
        #  consider the case where the normal plane at the s_position intersects a profile (in which case we need
        #  to interpolate half to the left and half to the right.



    def _find_type_positions(self, s_start: float, s_end: float, line_name: str) -> List[TypePosition]:
        type_bounds = self._type_bounds()

        bound_idx_start = bisect.bisect_right(type_bounds, s_start, key=lambda bound: bound[0]) - 1
        bound_idx_end = bisect.bisect_left(type_bounds, s_end, lo=bound_idx_start, key=lambda bound: bound[1]) + 1

        type_positions = [bound[2] for bound in type_bounds[bound_idx_start:bound_idx_end] if bound[2] is not None]
        return type_positions


    @lru_cache
    def _type_bounds(self) -> List[Tuple[float, float, Optional[TypePosition]]]:
        """Compute the bounds of each aperture type along the line.

        Returns
        -------
        type_bounds
            List of tuples ``(s_start, s_end, type_position)`` where each entry
            corresponds to a unique occurrence of a type position along the
            line ``line_name``. The entries are sorted and contiguous, and if for
            some range ``(s_start, s_end)`` there is no associated type_position
            (i.e. there's a gap in the aperture model), ``type_position`` is None.
        """
        line_name = self.aperture_model.line_name
        line = self.env[line_name]
        survey = line.survey()
        line_length = survey.s[-1]

        type_positions = list(self.aperture_model.type_positions.positions)
        if not type_positions:
            return [(0.0, line_length, None)]

        ref_s_list = []
        type_ranges = []

        for type_pos in type_positions:
            aperture_type = self.type_for_position(type_pos)
            positions = list(aperture_type.positions)

            if not positions:
                print(f"Warning: aperture type {self.type_name_for_position(type_pos)} has no profile positions.")
                continue

            s_positions = [float(p.s_position) for p in positions]
            if any(s_positions[i] > s_positions[i + 1] for i in range(len(s_positions) - 1)):
                raise ValueError(
                    f"Profile positions are not ordered for the type relative to {type_pos.ref_position} "
                    f"(type index {type_pos.type_index})."
                )

            min_local = s_positions[0]
            max_local = s_positions[-1]

            sv_point = survey.rows[type_pos.idx_position]
            ref_s = sv_point.s[0]
            # TODO: include the transformation at some point

            ref_s_list.append(ref_s)
            type_ranges.append((ref_s + min_local, ref_s + max_local, type_pos))

        for i in range(len(ref_s_list) - 1):
            if ref_s_list[i] > ref_s_list[i + 1]:
                raise ValueError("Type positions are not ordered by increasing s.")

        type_bounds = []
        current_s = 0.0

        for start, end, type_pos in type_ranges:
            if current_s - start > self.s_tol:
                raise ValueError(f"Overlapping aperture types detected. Previous ends at {current_s}, current starts at {start}.")

            if start - current_s > self.s_tol:
                type_bounds.append((current_s, start, None))

            type_bounds.append((start, end, type_pos))
            current_s = end

        if current_s > line_length:
            raise ValueError("Aperture type bounds exceed line length.")

        if current_s < line_length:
            type_bounds.append((current_s, line_length, None))

        return type_bounds

    @singledispatchmethod
    @staticmethod
    def _profile_from_limit_element(element: LimitTypes) -> Tuple[ProfileTypes, float, float]:
        """
        Convert a limit beam element to a profile object.

        Parameters
        ----------
        element: LimitTypes
            Element to convert to a profile.
        Returns:
            A tuple consting of the profile type, x offset, and y offset.
        """
        raise NotImplementedError(f"Unsupported element type: {type(element)}")

    @_profile_from_limit_element.register
    @staticmethod
    def _profile_from_limit_rect(element: apertures.LimitRect) -> Tuple[ProfileTypes, float, float]:
        half_width = (element.max_x - element.min_x) / 2
        half_height = (element.max_y - element.min_y) / 2
        x = (element.min_x + element.max_x) / 2
        y = (element.min_y + element.max_y) / 2
        rectangle = Rectangle(half_width=half_width, half_height=half_height)
        return rectangle, x, y

    @_profile_from_limit_element.register
    @staticmethod
    def _profile_from_limit_ellipse(element: apertures.LimitEllipse) -> Tuple[ProfileTypes, float, float]:
        rx = element.a
        ry = element.b
        ellipse = Ellipse(half_major=rx, half_minor=ry)
        return ellipse, 0, 0

    @_profile_from_limit_element.register
    @staticmethod
    def _profile_from_limit_rect_ellipse(element: apertures.LimitRectEllipse) -> Tuple[ProfileTypes, float, float]:
        max_x = element.max_x
        max_y = element.max_y
        rx = element.a
        ry = element.b
        rect_ellipse = RectEllipse(max_x=max_x, max_y=max_y, half_major=rx, half_minor=ry)
        return rect_ellipse, 0, 0

    @_profile_from_limit_element.register
    @staticmethod
    def _profile_from_limit_racetrack(element: apertures.LimitRacetrack) -> Tuple[ProfileTypes, float, float]:
        half_width = (element.max_x - element.min_x) / 2
        half_height = (element.max_y - element.min_y) / 2
        x = (element.min_x + element.max_x) / 2
        y = (element.min_y + element.max_y) / 2
        rx = element.a
        ry = element.b
        racetrack = Racetrack(
            half_width=half_width,
            half_height=half_height,
            half_major=rx,
            half_minor=ry,
        )
        return racetrack, x, y

    @_profile_from_limit_element.register
    @staticmethod
    def _profile_from_limit_polygon(element: apertures.LimitPolygon) -> Tuple[ProfileTypes, float, float]:
        xs = element.x_vertices + [element.x_vertices[0]]
        ys = element.y_vertices + [element.y_vertices[0]]
        polygon = Polygon(vertices=np.column_stack([xs, ys]))
        return polygon, 0, 0

    @classmethod
    def _profile_from_madx_aperture(cls, shape: str, params: List[float]) -> Optional[ProfileTypes]:
        converter, allowed_len_params = {
            'circle': (cls._profile_from_madx_circle, 1),
            'rectangle': (cls._profile_from_madx_rectangle, 2),
            'ellipse': (cls._profile_from_madx_ellipse, 2),
            'rectellipse': (cls._profile_from_madx_rectellipse, 4),
            'racetrack': (cls._profile_from_madx_racetrack, 4),
            'octagon': (cls._profile_from_madx_octagon, 4),
        }[shape]

        # Clean up params due to MAD-X quirks
        params = params[:allowed_len_params]

        if np.any(np.array(params[allowed_len_params:]) != 0):
            raise ValueError(
                f"Extra non-zero parameters provided for MAD-X aperture shape "
                f"{shape}. Accepted number of params is {allowed_len_params}; "
                f"provided {params}."
            )

        # If all params are zero, we ignore the aperture
        if np.all(params == 0):
            return None

        return converter(*params)

    @staticmethod
    def _profile_from_madx_circle(radius) -> Circle:
        return Circle(radius=radius)

    @staticmethod
    def _profile_from_madx_rectangle(half_width, half_height) -> Rectangle:
        return Rectangle(half_width=half_width, half_height=half_height)

    @staticmethod
    def _profile_from_madx_ellipse(half_major, half_minor) -> Ellipse:
        return Ellipse(half_major=half_major, half_minor=half_minor)

    @staticmethod
    def _profile_from_madx_rectellipse(max_x, max_y, half_major, half_minor) -> RectEllipse:
        return RectEllipse(
            max_x=max_x,
            max_y=max_y,
            half_major=half_major,
            half_minor=half_minor,
        )

    @staticmethod
    def _profile_from_madx_racetrack(half_width, half_height, half_major, half_minor) -> Racetrack:
        return Racetrack(
            half_width=half_width,
            half_height=half_height,
            half_major=half_major,
            half_minor=half_minor,
        )

    @staticmethod
    def _profile_from_madx_octagon(half_width, half_height, angle_0, angle_1) -> Octagon:
        # TODO: Handle inconsistencies coming from angle_1
        x = 0.5 * half_width * (np.tan(angle_0) + 1)
        diag = np.sqrt(2) * x
        return Octagon(half_width=half_width, half_height=half_height, half_diagonal=diag)

    @classmethod
    def _guess_original_mad_name(cls, element_name) -> Any:
        """Given a name of an element in a line, de-mangle the original MAD-X name.

        When importing a line from MAD-X, names can be mangled in two ways:
        1. ``:N`` may be added for the (N+1)-th repetition of the same element.
        2. ``/line_name`` may be appended if the same element appears in multiple sequences.
        This function only works if the elements were not named according to these patterns by the user,
        and as such is a bit of a hack. In corner cases it is best not to go through cpymad,
        but instead use the native loader with ``Aperture.from_line_with_associated_apertures``.

        Parameters
        ----------
        element_name : str
            Name of a beam element.
        """
        pattern = r"(?P<prefix>.*?)(?:[:]\d+)?(?:/[^/]+)?"
        match = re.fullmatch(pattern, element_name)
        return match.group('prefix')
