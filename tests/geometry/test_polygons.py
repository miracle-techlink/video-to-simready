"""Direct coverage for the boundary-inclusive convex polygon test."""

import numpy as np

from simready.geometry.polygons import inside_convex_polygon


SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_inside_convex_polygon_basic_membership():
    assert inside_convex_polygon([0.5, 0.5], SQUARE)
    assert not inside_convex_polygon([1.5, 0.5], SQUARE)
    assert not inside_convex_polygon([-0.1, 0.5], SQUARE)


def test_inside_convex_polygon_includes_boundary():
    for point in ([0.0, 0.0], [1.0, 1.0], [0.5, 0.0], [0.0, 0.5]):
        assert inside_convex_polygon(point, SQUARE)


def test_inside_convex_polygon_independent_of_winding():
    reversed_square = list(reversed(SQUARE))
    for point in ([0.5, 0.5], [1.5, 0.5], [0.5, -0.5]):
        assert inside_convex_polygon(point, SQUARE) == inside_convex_polygon(
            point, reversed_square
        )


def test_inside_convex_polygon_triangle():
    triangle = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]
    assert inside_convex_polygon([1.0, 0.5], triangle)
    assert not inside_convex_polygon([3.0, 2.0], triangle)


def test_inside_convex_polygon_degenerate_single_point_is_boundary_inclusive():
    # A degenerate polygon has zero-length edges, so every cross product is zero
    # and the point is reported as inside. Document the boundary behavior.
    assert inside_convex_polygon([1.0, 1.0], [[1.0, 1.0]])
    assert inside_convex_polygon([0.0, 0.0], [[1.0, 1.0]])


def test_inside_convex_polygon_degenerate_line():
    # Points on the segment qualify; off-segment points do not.
    segment = [[0.0, 0.0], [2.0, 0.0]]
    assert inside_convex_polygon([1.0, 0.0], segment)
    assert not inside_convex_polygon([1.0, 0.1], segment)


def test_inside_convex_polygon_nonconvex_shape_gives_wrong_answer_documented():
    # inside_convex_polygon assumes convexity; for a concave polygon a point in
    # the notch is classified by the convex-hull semantics of the cross-product
    # sweep. This documents current behavior rather than endorsing it.
    concave = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0]]
    # Point clearly outside the concave polygon's left arm region.
    assert not inside_convex_polygon([-1.0, 1.0], concave)


def test_inside_convex_polygon_accepts_numpy_arrays():
    assert inside_convex_polygon(np.array([0.5, 0.5]), np.array(SQUARE))
