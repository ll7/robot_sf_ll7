from math import cos, sin, pi
import numpy as np
from pytest import approx
from robot_sf.geometry import cos_sim, euclid_dist


def test_dist_to_point_itself_is_zero():
    x, y = np.random.uniform(size=(2))
    assert euclid_dist((x, y), (x, y)) == 0


def test_dist_in_unit_circle_is_one():
    angle = np.random.uniform(0, 2*pi, size=(1))
    x, y = cos(angle), sin(angle)
    assert euclid_dist((0, 0), (x, y)) == approx(1)


def test_dist_in_any_circle_is_the_radius():
    radius = np.random.uniform(1, 10, size=(1))
    angle = np.random.uniform(0, 2*pi, size=(1))
    x, y = cos(angle) * radius, sin(angle) * radius
    assert euclid_dist((0, 0), (x, y)) == approx(radius)


def test_similarity_of_aligned_vectors_is_one():
    scale = np.random.uniform(0, 10, size=(1))
    v1 = np.random.uniform(0, 10, size=(2))
    v2 = v1 * scale
    assert cos_sim(v1, v2) == approx(1)


def test_similarity_of_contrary_vectors_is_minus_one():
    scale = np.random.uniform(0, 10, size=(1))
    v1 = np.random.uniform(0, 10, size=(2))
    v2 = v1 * scale * -1
    assert cos_sim(v1, v2) == approx(-1)


def test_similarity_of_orthogonal_vectors_is_zero():
    v1 = np.random.uniform(0, 10, size=(2))
    v2 = v1[1], v1[0] * -1
    assert cos_sim(v1, v2) == approx(0)
