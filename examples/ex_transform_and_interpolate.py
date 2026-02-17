import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
from cgeom.aperture import Aperture
from cgeom.structures import ApertureModel, ApertureType, Circle, Profile, ProfilePosition, Rectangle, TypePosition


env = xt.Environment()

l = 1
dx = 1
angle = np.deg2rad(30)
l_straight = dx / np.sin(angle / 2)
rho = 0.5 * l_straight / np.sin(angle / 2)
l_curv = rho * angle

drift = env.new('drift', xt.Drift, length=l)
rot_plus = env.new('rot_plus', xt.Bend, length=l_curv, angle=angle, k0=0)
rot_minus = env.new('rot_minus', xt.Bend, length=l_curv, angle=-angle, k0=0)

line = env.new_line(
    name='line',
    components=[drift, rot_plus, drift, drift, rot_minus, drift],
)

sv = line.survey()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(sv.Z, sv.X, sv.Y, c='b')
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')
ax.set_aspect('equal', 'datalim')
plt.show()

circle = Circle(radius=1)
rectangle = Rectangle(half_width=1, half_height=1.5)

profiles = [
    Profile(shape=circle, tol_r=0, tol_x=0, tol_y=0),
    Profile(shape=rectangle, tol_r=0, tol_x=0, tol_y=0),
]

profile_positions = [
    ProfilePosition(profile_index=0, s_position=s)
    for s in np.linspace(0, 11, 12)
]

types = [
    ApertureType(curvature=0, positions=[0, 1]),
]

type_positions = [
    TypePosition(
        type_index=0,
        survey_reference_name='drift::0',
        survey_index=0,
        transformation=np.identity(4),
    ),
]

model = ApertureType(type_positions=type_positions, types=types, profiles=profiles)

aper = ApertureModel(
    line_name='line',
    type_positions=type_positions,
    types=types,
    profiles=profiles,
    type_names=['type0'],
    profile_names=['circle', 'rectangle'],
)
