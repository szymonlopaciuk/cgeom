import numpy as np
from cgeom import Path2D, clib


def test_is_point_inside_polygon_ellipse():
    rx = 2
    ry = 3
    ellipse = Path2D.from_ellipse(rx=2, ry=3)
    ellipse_poly = ellipse.get_points(ds_min=0.01)

    @np.vectorize
    def in_ellipse(x, y):
        point = clib.G2DPoint(x=x, y=y)
        return clib.geom2d_is_point_inside_polygon(point, ellipse_poly)

    extent = np.linspace(-10, 10, 100)
    xs, ys = np.meshgrid(extent, extent)

    result = in_ellipse(xs, ys).view(bool)
    expected = (xs ** 2 / rx ** 2 + ys ** 2 / ry ** 2 - 1) < 0

    assert not np.all(result) and np.any(result)  # sanity check
    assert np.all(result == expected)


def test_is_point_inside_polygon_path():
    # Define a shape that is a rectangle spanning (-1, -1) through (3, 2) minus
    # a rectangle (1, 0.5) through (2, 2)

    poly = [
        (1, 2),
        (1, .5),
        (2, .5),
        (2, 2),
        (3, 2),
        (3, -1),
        (-1, -1),
        (-1, 2),
        (1, 2),
    ]
    points = np.array(poly, dtype=clib.G2DPoint.dtype)

    @np.vectorize
    def in_poly(x, y):
        point = clib.G2DPoint(x=x, y=y)
        return clib.geom2d_is_point_inside_polygon(point, points)

    extent = np.linspace(-5, 5, 100)
    xs, ys = np.meshgrid(extent, extent)

    result = in_poly(xs, ys).view(bool)

    in_rec1 = (-1 < xs) & (xs < 3) & (-1 < ys) & (ys < 2)
    in_rec2 = (1 < xs) & (xs < 2) & (0.5 < ys) & (ys < 2)
    expected = in_rec1 & ~in_rec2

    assert not np.all(result) and np.any(result)  # sanity check
    assert np.all(result == expected)


def test_points_inside_polygon_inscribed_circles():
    r1 = 0.11
    r2 = 1

    circ1 = [(r1 * np.cos(angle), r1 * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 99)]
    circ1.append(circ1[0])
    circ2 = [(r2 * np.cos(angle), r2 * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 99)]
    circ2.append(circ2[0])

    circ1 = np.array(circ1, dtype=clib.G2DPoint.dtype)
    circ2 = np.array(circ2, dtype=clib.G2DPoint.dtype)

    small_in_big = clib.geom2d_points_inside_polygon(circ1, circ2)
    assert int.from_bytes(small_in_big) == 1

    big_in_small = clib.geom2d_points_inside_polygon(circ2, circ1)
    assert int.from_bytes(big_in_small) == 0


def test_points_inside_polygon_simple():
    poly_big = [(1, 1), (2, 3.5), (4.5, 3.5), (4.5, 1), (1, 1)]
    poly_small = [(2, 2), (3, 3), (4, 2), (3, 1.5), (2, 2)]

    poly_big = np.array(poly_big, dtype=clib.G2DPoint.dtype)
    poly_small = np.array(poly_small, dtype=clib.G2DPoint.dtype)

    small_in_big = clib.geom2d_points_inside_polygon(poly_small, poly_big)
    assert int.from_bytes(small_in_big) == 1

    big_in_small = clib.geom2d_points_inside_polygon(poly_big, poly_small)
    assert int.from_bytes(big_in_small) == 0


def test_points_inside_polygon_simpler():
    poly_small = [(-1, -1), (0, 1), (1, -1), (-1, -1)]
    poly_big = [(-2, -2), (0, 2), (2, -2), (-2, -2)]

    poly_big = np.array(poly_big, dtype=clib.G2DPoint.dtype)
    poly_small = np.array(poly_small, dtype=clib.G2DPoint.dtype)

    small_in_big = clib.geom2d_points_inside_polygon(poly_small, poly_big)
    assert int.from_bytes(small_in_big) == 1

    big_in_small = clib.geom2d_points_inside_polygon(poly_big, poly_small)
    assert int.from_bytes(big_in_small) == 0
