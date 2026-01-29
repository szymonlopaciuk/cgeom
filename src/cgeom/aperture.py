import bisect
from functools import singledispatchmethod, lru_cache
from typing import Dict, Union, Tuple, List, Optional, Iterator

import numpy as np

import xobjects as xo
from xtrack.beam_elements import apertures
from xtrack.progress_indicator import progress


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


class ProfileUnion(xo.UnionRef):
    _reftypes = [Circle, Rectangle, Ellipse, RectEllipse, Racetrack, Octagon, Polygon, SVGShape]


class ProfileList(xo.Struct):
    profiles = ProfileUnion[:]


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
        self.indices = indices  # dict of profile names to indices
        self.profile_list = profiles  # list of profile objects

    def __getitem__(self, name: str) -> ProfileTypes:
        try:
            index = self.indices[name]
            return self.profile_list.profiles[index]
        except KeyError:
            raise KeyError(f"Profile {name} not found.")


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
    names: dict
        Dictionary mapping names of types to indices in ``types``.
    types: ApertureTypeList
        List of pipes in the lab frame.
    """
    def __init__(self, indices: Dict[str, int], types: ApertureTypeList):
        self.indices = indices
        self.types = types


class TypePosition(xo.Struct):
    type_index = xo.Int32
    ref_position = xo.String  # identify a point in survey
    idx_position = xo.Int32  # index of the point in the survey
    transformation = xo.Float32[
        4, 4
    ]  # 3D rigid transformation matrix from the ref point to the center of the aperture type


class TypePositionList(xo.Struct):
    positions = TypePosition[:]


class ApertureModel:
    def __init__(self, indices: Dict[str, int], line_name: str, type_positions: TypePositionList):
        self.indices = indices  # dict of aperture model names to indices
        self.line_name = line_name
        self.type_positions = type_positions  # positioning of types in line frame


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
    ):
        self.env = env
        self.profiles = profiles  # list of profile objects
        self.aperture_types = aperture_types  # list of pipes in the lab frame
        self.aperture_model = aperture_model  # positioning of types in line frame
        self.cross_sections = cross_sections
        self.halo_params = self.halo_params.copy()
        if halo_params is not None:
            self.halo_params.update(halo_params)

    @classmethod
    def from_line_with_aperture(cls, line, line_name=None):
        survey = line.survey()
        name_to_sv_index = dict(zip(survey.name, range(len(survey))))
        layout_data = line.metadata['layout_data']

        element_for_aperture = {}

        name_iter_with_progress = progress(
            line.element_names,
            desc="Collecting apertures",
            total=len(line.element_names),
        )

        for name in name_iter_with_progress:
            # Discard line name suffix to get the aperture name
            aper_name = name.rsplit("/", 1)[0] if "/" in name else name
            if aper_name in layout_data:
                element_for_aperture[aper_name] = name

        aper_iter_with_progress = progress(
            element_for_aperture.items(),
            desc="Building aperture data",
            total=len(element_for_aperture),
        )

        profiles = []
        type_list = []
        type_positions_list = []
        aper_names = {}

        aper_idx = 0

        for aper_name, element_name in aper_iter_with_progress:
            element_metadata = layout_data[aper_name]

            if 'aperture' not in element_metadata:
                continue

            shape, params, tols = element_metadata['aperture']
            profile = cls._profile_from_madx_aperture(shape, params)

            if not profile:
                # There is not really an aperture here, continue
                continue

            element = line.element_dict[element_name]
            aper_names[aper_name] = aper_idx

            profile_union = ProfileUnion(profile)
            profiles.append(profile_union)

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
                profile_position_end.s_position = element.length / 2
                positions = [profile_position_start, profile_position_end]
                curvature = getattr(element, 'h', 0)
            else:
                # Place single profile at center of element
                positions = [profile_position]
                curvature = 0

            aperture_type = ApertureType(curvature=curvature, positions=positions)
            type_list.append(aperture_type)

            # Apply element transformations to type
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
                type_index=aper_idx,
                ref_position=element_name,
                idx_position=name_to_sv_index[element_name],
                transformation=matrix,
            )
            type_positions_list.append(type_position)

            aper_idx += 1

        profile_list = ProfileList(profiles=profiles)
        profiles = Profiles(indices=aper_names, profiles=profile_list)

        type_positions = TypePositionList(positions=type_positions_list)

        model = ApertureModel(
            indices=aper_names,
            line_name=line_name or line.name,
            type_positions=type_positions,
        )

        types = ApertureTypes(
            indices=aper_names,
            types=ApertureTypeList(types=type_list),
        )

        aperture = cls(
            env=line.env,
            profiles=profiles,
            aperture_types=types,
            aperture_model=model,
            cross_sections=None,
        )

        return aperture

    @classmethod
    def from_line_with_limit(cls, line, line_name=None):
        survey = line.survey()
        name_to_sv_index = dict(zip(survey.name, range(len(survey))))
        element_names = line.element_names

        profiles = []
        type_list = []
        indices = {}
        type_positions_list = []

        aper_idx = 0

        for name in progress(element_names, desc="Building aperture data", total=len(element_names)):
            element = line.element_dict[name]
            if not isinstance(element, LimitTypes):
                continue

            indices[name] = aper_idx
            profile, center_x, center_y = cls._profile_from_limit_element(element)
            profile_union = ProfileUnion(profile)
            profiles.append(profile_union)

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

        profile_list = ProfileList(profiles=profiles)
        profiles = Profiles(indices=indices, profiles=profile_list)

        type_positions = TypePositionList(positions=type_positions_list)

        model = ApertureModel(
            indices=indices,
            line_name=line_name or line.name,
            type_positions=type_positions,
        )

        types = ApertureTypes(
            indices=indices,
            types=ApertureTypeList(types=type_list),
        )

        aperture = cls(
            env=line.env,
            profiles=profiles,
            aperture_types=types,
            aperture_model=model,
            cross_sections=None,
        )

        return aperture

    def get_aperture_margin_mm(self, line_name: str, element_name: str) -> np.ndarray:
        line = self.env[line_name]
        element = line[element_name]
        s_start = line.get_s_position(element_name)
        s_end = s_start + getattr(element, 'length', 0)

        apertures = self._find_profiles(s_start, s_end, line_name)

        import ipdb; ipdb.set_trace()

    def _find_profiles(self, s_start: float, s_end: float, line_name: str) -> List[Tuple[TypePosition, ProfilePosition]]:
        sorted_profiles = self._sorted_profiles(line_name)

        idx_start = bisect.bisect_left(sorted_profiles, s_start, key=lambda p: p[0])
        idx_end = bisect.bisect_right(sorted_profiles, s_end, lo=idx_start, key=lambda p: p[1])

        profiles = [(type_pos, profile_pos) for (_, type_pos, profile_pos) in sorted_profiles[idx_start:idx_end]]
        return profiles

    @lru_cache
    def _sorted_profiles(self, line_name: str) -> List[Tuple[float, TypePosition, ProfilePosition]]:
        """Make a list of sorted profile positions based on their absolute positions in the line.

        Parameters
        ----------
        line_name: str
            Name of the line for which to make the sorted profile list.

        Returns
        -------
        A list of tuples ``(s_position, aperture_type, profile_position)``, where each entry corresponds to a unique
        occurrence of a profile position along the line ``line_name``, sorted by ``s_position``.
        """
        survey = self.env[line_name].survey()
        profiles = []

        for type_pos in self.aperture_model.type_positions.positions:
            aperture_type = self.aperture_types.types[type_pos.type_index]
            sv_point = survey.rows[type_pos.idx_position]

            for profile_pos in aperture_type.positions:
                # TODO: I think we need to take into account the transformations in this calculation...
                s_position = sv_point.s[0] + sv_point.length[0] / 2 + profile_pos.s_position

                profiles.append((s_position, type_pos, profile_pos))

        profiles = sorted(profiles, key=lambda p: p[0])
        import ipdb; ipdb.set_trace()
        return profiles

    @singledispatchmethod
    @staticmethod
    def _profile_from_limit_element(element: LimitTypes) -> Tuple[ProfileTypes, float, float]:
        """
        Convert a limit beam element to a profile object.

        Parameters
        ----------
        element: LimitTypes
            Element to convert to a profile.
        Returns: ProfileTypes
            A profile.
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
