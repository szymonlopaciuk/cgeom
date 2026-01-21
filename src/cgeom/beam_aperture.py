from typing import Callable, Dict, Union, Type, Tuple
from itertools import pairwise

import numpy as np

from xtrack.beam_elements import apertures
from xtrack.progress_indicator import progress

from . import clib
from .path import Path2D
from .aperture import transform_matrix


LimitTypes = Union[
    apertures.LimitRect,
    apertures.LimitEllipse,
    apertures.LimitRectEllipse,
    apertures.LimitRacetrack,
    apertures.LimitPolygon,
]


class Aperture:
    def __init__(self, line, tol_x, tol_y, tol_r):
        self.line = line
        self._twiss = line.twiss()
        self.beam_data = clib.G2DBeamData()

        self.tol_x = tol_x
        self.tol_y = tol_y
        self.tol_r = tol_r

        s_positions, twiss_data, apertures, types, name_to_index = self.build_aperture_data()
        self.s_positions = s_positions
        self.twiss_data = twiss_data
        self.apertures = apertures
        self.types = types
        self._name_to_index = name_to_index

    def set_beam_data(
            self,
            emitt_x_norm,
            emitt_y_norm,
            delta_rms,
            tol_co,
            tol_disp,
            tol_disp_ref_dx,
            tol_disp_ref_beta,
            tol_energy,
            tol_betabeating,
            halo_x,
            halo_y,
            halo_r,
            halo_primary,
    ):
        """Set beam attributes.

        Parameters
        ----------
        emitt_x_norm: float
            normalized emittance x
        emitt_y_norm: float
            normalized emittance y
        delta_rms: float
            rms energy spread
        tol_co: float
            tolerance for closed orbit
        tol_disp: float
            tolerance for normalized dispersion
        tol_disp_ref_dx: float
            tolerance for reference dispersion derivative
        tol_disp_ref_beta: float
            tolerance for reference dispersion beta
        tol_energy: float
            tolerance for energy error
        tol_betabeating: float
            tolerance for betabeating in sigma
        halo_x: float
            n sigma of horizontal halo
        halo_y: float
            n sigma of vertical halo
        halo_r: float
            n sigma of 45 degree halo
        halo_primary: float
            n sigma of primary halo
        """
        self.beam_data = clib.G2DBeamData(
            emitx_norm=emitt_x_norm,
            emity_norm=emitt_y_norm,
            delta_rms=delta_rms,
            tol_co=tol_co,
            tol_disp=tol_disp,
            tol_disp_ref_dx=tol_disp_ref_dx,
            tol_disp_ref_beta=tol_disp_ref_beta,
            tol_energy=tol_energy,
            tol_betabeating=tol_betabeating,
            halo_x=halo_x,
            halo_y=halo_y,
            halo_r=halo_r,
            halo_primary=halo_primary,
        )

    def get_s(self, name):
        return self.s_positions[self._name_to_index[name]]

    def get_twiss(self, name):
        return self.twiss_data[self._name_to_index[name]]

    def get_aperture(self, name):
        return self.apertures[self._name_to_index[name]]

    def build_aperture_data(self):
        line_table = self.line.get_table()
        s_positions, twiss_data, paths, types = [], [], [], []
        names = self.line.element_names
        name_to_index = {}

        for idx, name in progress(enumerate(names), desc="Building aperture data", total=len(names)):
            element = self.line.element_dict[name]
            if not isinstance(element, LimitTypes):
                continue

            aperture = self.get_aperture_data_from_limit_element(name)
            if element.transformations_active:
                self.apply_transform(aperture, element)
            twiss = self.get_twiss_data_at_element(idx)

            name_to_index[name] = len(s_positions)
            s_positions.append(line_table.s[idx])
            twiss_data.append(twiss)
            paths.append(aperture)
            types.append(type(element))

        return s_positions, twiss_data, paths, types, name_to_index

    def get_aperture_data_from_limit_element(self, name):
        """
        Get 2d points from limit element

        Parameters
        ----------
        name : str
            Name of the limit element
        Returns: G2DBeamApertureData
            2D beam aperture data
        """
        element: LimitTypes = self.line.element_dict[name]
        path, centre_x, centre_y = self.CONVERTER_FUNCTIONS[type(element)](element)
        points = clib.geom2d_path_get_points(path)
        points['x'] += centre_x
        points['y'] += centre_y
        beam_aperture = clib.G2DBeamApertureData(
            points=points,
            n_points=len(points),
            tol_r=self.tol_r,
            tol_x=self.tol_x,
            tol_y=self.tol_y,
        )
        return beam_aperture

    def get_twiss_data_at_element(self, name):
        """
        Get twiss data at element 
        """
        row = self._twiss.rows[name]
        c_twiss_data = clib.G2DTwissData(
            x=row['x'][0],
            y=row['y'][0],
            betx=row['betx'][0],
            bety=row['bety'][0],
            dx=row['dx'][0],
            dy=row['dy'][0],
            delta=row['delta'][0],
            gamma=row['gamma0'],
        )
        return c_twiss_data

    @staticmethod
    def get_path2d_from_limit_rect(element: apertures.LimitRect) -> Tuple[Path2D, float, float]:
        half_width = (element.max_x - element.min_x) / 2
        half_height = (element.max_y - element.min_y) / 2
        x = (element.min_x + element.max_x) / 2
        y = (element.min_y + element.max_y) / 2
        return Path2D.from_rectangle(half_width, half_height), x, y

    @staticmethod
    def get_path2d_from_limit_ellipse(element: apertures.LimitEllipse) -> Tuple[Path2D, float, float]:
        rx = element.a
        ry = element.b
        return Path2D.from_ellipse(rx, ry), 0, 0

    @staticmethod
    def get_path2d_from_limit_rect_ellipse(element: apertures.LimitRectEllipse) -> Tuple[Path2D, float, float]:
        max_x = element.max_x
        max_y = element.max_y
        rx = element.a
        ry = element.b
        return Path2D.from_rectellipse(max_x, max_y, rx, ry), 0, 0

    @staticmethod
    def get_path2d_from_limit_racetrack(element: apertures.LimitRacetrack) -> Tuple[Path2D, float, float]:
        half_width = (element.max_x - element.min_x) / 2
        half_height = (element.max_y - element.min_y) / 2
        x = (element.min_x + element.max_x) / 2
        y = (element.min_y + element.max_y) / 2
        rx = element.a
        ry = element.b
        return Path2D.from_racetrack(half_width, half_height, rx, ry), x, y

    @staticmethod
    def get_path2d_from_limit_polygon(element: apertures.LimitPolygon) -> Tuple[Path2D, float, float]:
        xs = element.x_vertices + [element.x_vertices[0]]
        ys = element.y_vertices + [element.y_vertices[0]]
        segments = []
        for (x0, y0), (x1, y1) in pairwise(zip(xs, ys)):
            segment = clib.geom2d_line_segment_from_start_end(x0, y0, x1, y1)
            segments.append(segment)

        return Path2D(segments, len_segments=len(segments))

    def apply_transform(self, aperture, element):
        points = aperture.points
        matrix = transform_matrix(
            dx=element.shift_x,
            dy=element.shift_y,
            ds=element.shift_s,
            theta=element.rot_y_rad,
            phi=element.rot_x_rad,
            psi=element.rot_s_rad_no_frame,
        )
        point_vecs = np.vstack([
            points['x'],
            points['y'],
            np.zeros(len(points)),
            np.ones(len(points)),
        ])
        transformed_points = matrix @ point_vecs

        # Project back on 2D plane
        points['x'] = transformed_points[0]
        points['y'] = transformed_points[1]

    @property
    def extents(self):
        for aperture in self.apertures:
            points = aperture.points
            yield max(points['x']), min(points['y']), min(points['x']), max(points['y'])

    @property
    def extents_on_axes(self):
        for aperture in self.apertures:
            points = aperture.points
            xs, ys = self._find_axis_intersections(points['x'], points['y'])
            yield xs[-1], ys[0], xs[0], ys[-1]

    @staticmethod
    def _find_axis_intersections(x, y):
        """Finds intersections of a closed polygon with x = 0 and y = 0.

        Returns:
            x_axis_points: sorted array of x-values where polygon crosses y = 0
            y_axis_points: sorted array of y-values where polygon crosses x = 0
        """
        # X Axis
        mask_xaxis = (y[:-1] * y[1:] <= 0) & (y[:-1] != y[1:])

        t_xaxis = y[:-1][mask_xaxis] / (
                    y[:-1][mask_xaxis] - y[1:][mask_xaxis])
        x_xaxis = x[:-1][mask_xaxis] + t_xaxis * (
                    x[1:][mask_xaxis] - x[:-1][mask_xaxis])

        # Y Axis
        mask_yaxis = (x[:-1] * x[1:] <= 0) & (x[:-1] != x[1:])

        t_yaxis = x[:-1][mask_yaxis] / (
                    x[:-1][mask_yaxis] - x[1:][mask_yaxis])
        y_yaxis = y[:-1][mask_yaxis] + t_yaxis * (
                    y[1:][mask_yaxis] - y[:-1][mask_yaxis])

        return np.sort(x_xaxis), np.sort(y_yaxis)

    CONVERTER_FUNCTIONS: Dict[Type[LimitTypes], Callable[[LimitTypes], Path2D]] = {
        apertures.LimitRect: get_path2d_from_limit_rect,
        apertures.LimitEllipse: get_path2d_from_limit_ellipse,
        apertures.LimitRectEllipse: get_path2d_from_limit_rect_ellipse,
        apertures.LimitRacetrack: get_path2d_from_limit_racetrack,
        apertures.LimitPolygon: get_path2d_from_limit_polygon,
    }
