from typing import Tuple

Vec2D = Tuple[float, float]
Line2D = Tuple[float, float, float, float]
Rect = Tuple[Vec2D, Vec2D, Vec2D]
# TODO: Is there a difference between a Rect and a Zone?
# rect ABC with sides |A B|, |B C| and diagonal |A C|
Zone = Tuple[Vec2D, Vec2D, Vec2D]
