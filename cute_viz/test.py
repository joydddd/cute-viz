from typing import Any


import numpy as np
from cutlass import cute
import cutlass
import itertools

@cute.jit
def flatten_to_tuple(shape: cute.Shape) -> tuple:
    return cute.flatten_to_tuple(shape)

@cute.jit
def unflatten(linear_coord: int, shape: cute.Shape) -> tuple:
    return cute.unflatten(linear_coord, shape)

@cute.jit
def apply_layout(layout: cute.Layout, coord: cute.Coord) -> int:
    return layout(coord)

def test(shape: cute.Shape, layout: cute.Layout) -> cute.Layout:

    # Get the size of each mode and concatenate
    r = cute.rank(layout)
    coord_space = tuple(cute.size(layout[i]) for i in range(r))
    print(coord_space)
        


@cute.jit
def test_shapes():
    shape = 10
    test(shape, cute.make_layout(shape))
    shape = (10,)
    test(shape, cute.make_layout(shape))
    shape = (2, 3)
    test(shape, cute.make_layout(shape))
    shape = (2, (2, 3))
    test(shape, cute.make_layout(shape))
    shape = (2, (2, (2, 3)))
    test(shape, cute.make_layout(shape))

test_shapes()