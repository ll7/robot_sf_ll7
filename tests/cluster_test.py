from functools import reduce
import numpy as np
from robot_sf.extenders_py_sf.clustering import k_means, find_outliers


def test_can_find_clusters():
    # np.random.seed(42)
    k, num_vecs, num_features = 5, 20, 2
    vecs = np.random.uniform(-10, 10, (num_vecs, num_features))
    clusters, centroids = k_means(vecs, k, 10000)

    assert sum([len(c) for c in clusters]) == num_vecs
    assert len(reduce(set.union, clusters).intersection(range(num_vecs))) == num_vecs
    # TODO: think of good assertions to measure distance


def test_find_outliers():
    # np.random.seed(42)
    num_vecs, num_features = 20, 2
    vecs = np.random.uniform(-1, 1, (num_vecs, num_features))
    vecs_with_outlier = np.concatenate((vecs, [[20.0, 20.0]]))

    assert find_outliers(vecs) == []
    assert find_outliers(vecs_with_outlier) == [num_vecs]
