from typing import List, Tuple
import numpy as np

Vec2D = Tuple[float, float]
Zone = Tuple[Vec2D, Vec2D, Vec2D] # rect ABC with sides |A B|, |B C| and diagonal |A C|


def sample_zone(zone: Zone, num_samples: int) -> List[Vec2D]:
    a, b, c = zone
    a, b, c = np.array(a), np.array(b), np.array(c)
    vec_ba, vec_bc = a - b, c - b
    rel_width = np.random.uniform(0, 1, (num_samples, 1))
    rel_height = np.random.uniform(0, 1, (num_samples, 1))
    points = b + rel_width * vec_ba + rel_height * vec_bc
    return [(x, y) for x, y in points]
