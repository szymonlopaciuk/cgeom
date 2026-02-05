from typing import List, Union, get_args

import xobjects as xo


class Circle(xo.Struct):
    radius = xo.Float32


class Rectangle(xo.Struct):
    half_width = xo.Float32
    half_height = xo.Float32


class Ellipse(xo.Struct):
    half_major = xo.Float32
    half_minor = xo.Float32


class RectEllipse(xo.Struct):
    half_width = xo.Float32
    half_height = xo.Float32
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
    _reftypes = get_args(ProfileTypes)


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
    profile_index = xo.Int32
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


class TypePosition(xo.Struct):
    type_index = xo.Int32
    survey_reference_name = xo.String  # identify a point in survey
    survey_index = xo.Int32  # index of the point in the survey
    transformation = xo.Float32[4, 4]  # 3D rigid transformation matrix from the survey entry to 0 s-position of type


class ApertureModel(xo.Struct):
    type_positions = TypePosition[:]
    types = ApertureType[:]
    profiles = Profile[:]

    def __init__(
        self,
        line_name: str,
        type_positions: List[TypePosition],
        types: List[ApertureType],
        profiles: List[ProfileTypes],
        type_names: List[str],
        profile_names: List[str],
        **kwargs,
    ):
        self.line_name = line_name

        if len(type_names) != len(types):
            raise ValueError("Length of type_names and type_names must match.")

        if len(profile_names) != len(profiles):
            raise ValueError("Length of profiles and profiles must match.")

        self.type_names = type_names
        self.profile_names = profile_names

        super().__init__(type_positions=type_positions, types=types, profiles=profiles, **kwargs)

    def type_name_for_index(self, idx: int) -> str:
        return self.type_names[idx]

    def profile_name_for_index(self, idx: int) -> str:
        return self.profile_names[idx]


class CrossSections(xo.Struct):
    count = xo.UInt32
    num_points = xo.UInt32
    s_positions = xo.Float32[:]
    type_position_indices = xo.UInt32[:]
    profile_position_indices = xo.UInt32[:]
    points = xo.Float32[:, :, 2]
