"""
Core visualization functions for CuTe layouts.
"""

import colorsys
import itertools
import numpy as np
import svgwrite
from typing import Any, Tuple, overload
from cutlass import cute, range_constexpr
from cutlass.cute import size, cosize, rank, make_identity_tensor, depth


# Hue palette: Different colors evenly distributed around the color wheel
# Values in degrees (0-360)
HUE_PALETTE = [
    0,      # Red
    45,     # Orange
    90,     # Yellow-green
    135,    # Green
    180,    # Cyan
    225,    # Blue
    270,    # Purple
    315,    # Magenta
]

# Lightness + Saturation palette: Controls color density/brightness
# Format: (Saturation 0-1, Lightness 0-1)
# Provides 8 levels of density from dark/saturated to light/desaturated
LIGHTNESS_SATURATION_PALETTE_COLOR = [
    (0.9, 0.50),    # Level 0: Very saturated, dark
    (0.9, 0.60),    # Level 1: Very saturated, medium-dark
    (0.8, 0.65),    # Level 2: High saturation, medium
    (0.7, 0.70),    # Level 3: Medium-high saturation
    (0.7, 0.75),    # Level 4: Medium saturation, medium-light
    (0.6, 0.80),    # Level 5: Medium-low saturation, light
    (0.5, 0.85),    # Level 6: Low saturation, very light
    (0.4, 0.90),    # Level 7: Desaturated, extremely light
]

# Select which palette to use
LIGHTNESS_SATURATION_PALETTE = LIGHTNESS_SATURATION_PALETTE_COLOR


def _create_color_scale(num_steps, hue=0, saturation=0.0, lightness_range=(0.0, 1.0), 
                        saturation_range=None, vary_saturation=False):
    """
    Create a color scale with specified number of steps.
    
    This function generates a gradient of colors similar to greyscale but with
    customizable hue and saturation. Colors transition from dark to light.
    Can create sophisticated gradients where saturation varies with lightness.
    
    Args:
        num_steps: Number of color steps in the scale
        hue: Hue value in degrees (0-360). Default 0 (red).
                For greyscale, hue doesn't matter when saturation=0
        saturation: Saturation level (0-1). Default 0.0 for greyscale.
                    Used as constant saturation, or max saturation if vary_saturation=True
        lightness_range: Tuple (min, max) for lightness values (0-1).
                        Default (0.0, 1.0) goes from black to white
        saturation_range: Tuple (min, max) for saturation values (0-1).
                         If None and vary_saturation=True, uses (saturation, saturation*0.4)
        vary_saturation: If True, saturation decreases as lightness increases,
                        creating more natural-looking gradients. Default False.
    
    Returns:
        List of (hue, saturation, lightness) tuples in HSL format
    
    Examples:
        # Greyscale (black to white)
        greyscale = _create_color_scale(10, saturation=0.0)
        
        # Blue gradient with constant saturation
        blue_scale = _create_color_scale(10, hue=225, saturation=0.8)
        
        # Blue gradient with varying saturation (more natural)
        blue_natural = _create_color_scale(10, hue=225, saturation=0.9, vary_saturation=True)
        
        # Custom saturation range
        custom = _create_color_scale(10, hue=0, saturation_range=(0.9, 0.4), 
                                     lightness_range=(0.5, 0.9), vary_saturation=True)
    """
    colors = []
    min_light, max_light = lightness_range
    
    # Determine saturation range
    if vary_saturation:
        if saturation_range is None:
            # Natural falloff: high saturation at dark end, lower at light end
            min_sat = saturation * 0.4
            max_sat = saturation
        else:
            min_sat, max_sat = saturation_range
    else:
        min_sat = max_sat = saturation
    
    colors = np.empty(num_steps, dtype=object)
    
    for idx in range(num_steps):
        # Calculate progress from 0 to 1
        if num_steps > 1:
            progress = idx / (num_steps - 1)
        else:
            progress = 0.5
        
        # Interpolate lightness (increases with progress)
        light = min_light + (max_light - min_light) * progress
        
        # Interpolate saturation (decreases with progress for natural look)
        sat = max_sat - (max_sat - min_sat) * progress
        
        colors[idx] = (hue, sat, light)
    
    return colors


def create_greyscale_palette(num_steps, reverse=False):
    """
    Create a greyscale color palette.
    
    Args:
        num_steps: Number of greyscale steps
        reverse: If True, goes from white to black. Default False (black to white)
    
    Returns:
        List of (hue, saturation, lightness) tuples in HSL format
        
    Examples:
        # Standard greyscale: black to white
        grey = _create_greyscale_palette(10)
        
        # Reversed: white to black
        grey_rev = _create_greyscale_palette(10, reverse=True)
    """
    if reverse:
        return _create_color_scale(num_steps, hue=0, saturation=0.0, lightness_range=(1.0, 0.0))
    else:
        return _create_color_scale(num_steps, hue=0, saturation=0.0, lightness_range=(0.0, 1.0))


def create_natural_palette(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Create a natural-looking color palette similar to LIGHTNESS_SATURATION_PALETTE_COLOR.
    
    This creates a sophisticated gradient where both saturation and lightness vary,
    producing natural-looking colors. When num_hues is provided, creates a
    product of hue × (saturation, lightness) pairs with hue as the outer dimension.
    Hues are evenly distributed around the color wheel.
    
    Args:
        shape: Either an int (total number of colors) or a tuple (num_steps, num_hues)
               - If int: returns 1D list with sqrt(shape) steps and calculated hues
               - If tuple: returns 2D list with shape[0] hues, each containing shape[1] colors
    
    Returns:
        - If shape is int: 1D list of (hue, saturation, lightness) tuples
        - If shape is tuple: 2D list where each sublist is one hue's color gradient
        
    Examples:
        # Single value generates a 1D list (e.g., 64 colors = 8x8 grid flattened)
        palette_64 = create_natural_palette(64)
        # Returns: [(hue0, sat0, light0), (hue0, sat1, light1), ...]  (1D list)
        
        # Tuple specifies (num_steps, num_hues) and returns 2D structure
        full_palette = create_natural_palette((8, 8))
        # Returns: [[(hue0, sat0, light0), ...], [(hue1, sat0, light0), ...], ...]  (2D list)
    """
    is_2d_color_platte = isinstance(shape, tuple)
    if is_2d_color_platte:
        num_steps, num_hues = shape
    else:
        num_steps = 8
        num_hues = (shape + num_steps - 1) // num_steps
    
    # Generate saturation/lightness pairs
    sat_light_pairs = _create_color_scale(
        num_steps if num_steps >= 8 else 8,
        hue=0,  # Placeholder, will be replaced
        saturation=0.9,  # Max saturation at dark end
        lightness_range=(0.50, 0.90),  # Dark to light range
        vary_saturation=True  # Saturation decreases as lightness increases
    )[:num_steps]
    

    hue_palette = [i * 360 / (num_hues if num_hues >= 8 else 8) for i in range(num_hues)]
    
    colors = np.empty((num_steps, num_hues), dtype=object)

    for j, hue in enumerate(hue_palette):
        for i, (_, sat, light) in enumerate(sat_light_pairs):
            colors[i, j] = (hue, sat, light)
    
    if is_2d_color_platte:
        return colors
    else:
        return colors.flatten(order='F')



# Combine hue and lightness/saturation palettes to create full HSL color space
# Format: (Hue in degrees 0-360, Saturation 0-1, Lightness 0-1)
# Generate dynamically using natural palette function with hue as outer dimension
HSL_COLORS = create_natural_palette(
    (len(LIGHTNESS_SATURATION_PALETTE), len(HUE_PALETTE))
)


# JIT-compiled helper functions
@cute.jit
def _flatten_to_tuple(shape):
    return cute.flatten_to_tuple(shape)

@cute.jit
def _unflatten_to_tuple(flatten_coord, shape):
    return cute.unflatten(flatten_coord, shape)


def _unflatten(flatten_coord, shape):
    if not isinstance(shape, tuple):
        coord, = flatten_coord
        return coord
    return _unflatten_to_tuple(flatten_coord, shape)


@cute.jit
def _apply_layout(layout: cute.Layout , coord: cute.Coord):
    return layout(coord)


@cute.jit
def _index_tensor(layout: cute.Tensor , coord: cute.Coord):
    return layout[coord]

@cute.jit
def _idx2crd(idx: int, shape: cute.Shape):
    return cute.idx2crd(idx, shape)


def _arange_shape(shape: cute.Shape) -> np.ndarray:
    coords = np.empty(size(shape), dtype=object)
    
    # Step 2: Iterate over all multi-dimensional coordinates
    for idx in range(size(shape)):
        coord = _idx2crd(idx, shape)
        coords[idx] = coord
    if isinstance(shape, tuple):
        target_shape = tuple[int, ...](size(dim) for dim in shape)
    else: 
        target_shape = shape
    return coords.reshape(target_shape, order='F')
    

def _extract_layout_indices(layout, coord_space):
    """
    Extract indices from a layout using compile-time loops.
    
    Allocates an array with shape = flatten(coord_space) and iterates over hierarchical coordinates.
    Each element is either an int (if target_space rank is 1) or a tuple (if target_space rank > 1).

    Args:
        layout: CuTe layout object
        coord_space: CuTe Shape object for the coordinate space (can be hierarchical)
                     Examples: 18, (3, 6), (3, (2, 3))
        target_space: CuTe Shape object for the target/indice space (optional)
                      If None, uses cosize(layout) as a 1D space

    Returns:
        N-dimensional numpy array with shape = flatten(coord_space)
        - If target_space rank is 1: elements are ints
        - If target_space rank > 1: elements are tuples of coordinates in target_space
        Examples:
        - coord_space=(3, 6), target_space=18 -> array of shape (3, 6) with int elements
        - coord_space=(3, 6), target_space=(4, 5) -> array of shape (3, 6) with tuple elements
    """
    coords_array = _arange_shape(coord_space)
    indices = np.empty_like(coords_array, dtype=object)
    
    for idx, coord in np.ndenumerate(coords_array):
        if isinstance(layout, cute.Tensor):
            indice = _index_tensor(layout, coord)
        else:
            indice = _apply_layout(layout, coord)
        indices[idx] = indice
    return indices


def _draw_layout_cell(dwg, x, y, cell_size, idx, colors, color_idx=None):
    """
    Draw a single layout cell with colored background and index label.
    
    Args:
        dwg: svgwrite.Drawing object
        x, y: Cell position
        cell_size: Size of cell
        idx: Index value to display (can be scalar or tuple for 2D indexing)
             - If tuple (i, j): displays first index on top line, second index on bottom line
             - If scalar: displays as single centered value
        colors: List of colors in HSL format (hue in degrees 0-360, saturation 0-1, lightness 0-1)
                Can be 1D list (for scalar color_idx) or 2D list (for tuple color_idx)
        color_idx: Optional index to use for coloring (defaults to idx if not provided)
                  Used to decouple display label from color selection
    
    Colors in HSL format are converted to RGB for rendering.
    """
    # Use idx for coloring if color_idx not provided
    if color_idx is None:
        color_idx = idx

    
    if rank(color_idx) == 1:
        hue, sat, light = colors.flatten(order='F')[color_idx % size(colors.shape)]
    else:
        assert rank(color_idx) == 2, "color_idx must be 1D or 2D"
        i, j = color_idx
        hue, sat, light = colors[i % len(colors)][j % len(colors[0])]
    
    rgb = tuple(int(c * 255) for c in colorsys.hls_to_rgb(hue / 360.0, light, sat))
    
    # Use white text for dark backgrounds (lightness < 0.5)
    text_color = "white" if light < 0.5 else "black"
    
    dwg.add(
        dwg.rect(
            insert=(x, y),
            size=(cell_size, cell_size),
            fill=svgwrite.rgb(*rgb, mode="RGB"),
            stroke="black",
        )
    )
    
    # Draw text labels (idx is used for display, not color_idx)
    if isinstance(idx, int):
        idx = (idx,)
    for r, i in enumerate(idx):
        dwg.add(
            dwg.text(
                str(f"{i},"),
                insert=(x + cell_size // 2, y + (r + 1/2) * (cell_size / len(idx))),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="8px",
                fill=text_color,
            )
        )


def _draw_axis_labels(dwg, positions):
    """
    Draw axis labels at specified positions.
    
    Args:
        dwg: svgwrite.Drawing object
        positions: List of (index, x, y) tuples for label positions
    """
    for i, x, y in positions:
        dwg.add(
            dwg.text(
                str(i),
                insert=(x, y),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="8px",
            )
        )

def _draw_hierarchical_separators(dwg, positions, span_length: float, orientation="horizontal", level=0):
    """
    Draw bold blue separator lines at specified positions.
    
    Args:
        dwg: svgwrite.Drawing object
        positions: List of (x, y) tuples where separators should be drawn
        span_length: Length of the separator line
        orientation: 'horizontal' or 'vertical'
    
    For horizontal lines: draws from (x, y) to (x + span_length, y)
    For vertical lines: draws from (x, y) to (x, y + span_length)
    """
    styles = [
        (3.0, 'blue'),      # Level 0: most important - very thick, emphasized color
        (2.0, 'steelblue'),     # Level 1: important
        (1.0, 'black'),        # Level 2+: baseline (matches cell borders)
    ]
    
    # Get style for this level (use last style if level exceeds array)
    style_index = min(level, len(styles) - 1)
    separator_width, separator_color = styles[style_index]
    
    for x, y in positions:
        if orientation == 'horizontal':
            # Horizontal line from (x, y) to (x + span_length, y)
            dwg.add(dwg.line(
                start=(x, y),
                end=(x + span_length, y),
                stroke=separator_color,
                stroke_width=separator_width
            ))
        elif orientation == 'vertical':
            # Vertical line from (x, y) to (x, y + span_length)
            dwg.add(dwg.line(
                start=(x, y),
                end=(x, y + span_length),
                stroke=separator_color,
                stroke_width=separator_width
            ))

def _create_1d_grid_svg(grid_shape, indices, color_palette, color_indices=None):
    """Create SVG for 1D grid visualization.
    
    Args:
        grid_shape: Shape of the grid to draw (int or tuple)
        indices: Array where indices[i] is the label for position i
        color_palette: List of (hue, saturation, lightness) tuples
        color_indices: Optional array where color_indices[i] is the color index for position i
                      If None, uses indices[i] for coloring
    
    Returns:
        svgwrite.Drawing object
    """
    cell_size = 20
    if color_indices is None:
        color_indices = indices
    
    M = size(grid_shape)
    
    # Create horizontal strip
    label_margin = cell_size
    page_width = M * cell_size + label_margin
    page_height = cell_size + label_margin
    
    dwg = svgwrite.Drawing(size=(page_width, page_height))
    
    # Draw cells
    for i in range(M):
        x = i * cell_size + label_margin
        y = label_margin
        _draw_layout_cell(dwg, x, y, cell_size, indices[i], color_palette, color_indices[i])
    
    # Add top axis labels
    label_positions = [
        (i, i * cell_size + label_margin + cell_size // 2, label_margin // 2)
        for i in range(M)
    ]
    _draw_axis_labels(dwg, label_positions)
    
    return dwg


def _create_2d_grid_svg(grid_shape, indices, color_palette, color_indices=None):
    """Create SVG for 2D grid visualization.
    
    Args:
        grid_shape: Shape of the grid to draw (tuple of (M, N))
                   Each mode can be hierarchical, e.g., ((2, 3), 3) for a 6×3 grid
        indices: Array where indices[i, j] is the label for position (i, j)
        color_palette: List of (hue, saturation, lightness) tuples
        color_indices: Optional array where color_indices[i, j] is the color index for position (i, j)
                      If None, uses indices[i, j] for coloring
    
    Returns:
        svgwrite.Drawing object
    """
    cell_size = 20

    if color_indices is None:
        color_indices = indices
    
    assert rank(grid_shape) == 2, "grid_shape must be a rank-2 tuple"

    M, N = size(grid_shape[0]), size(grid_shape[1])
    
    # Add margin for axis labels
    label_margin = cell_size
    page_width = N * cell_size + label_margin
    page_height = M * cell_size + label_margin

    dwg = svgwrite.Drawing(size=(page_width, page_height))

    # Draw grid cells
    for i in range(M):
        for j in range(N):
            x = j * cell_size + label_margin
            y = i * cell_size + label_margin
            _draw_layout_cell(dwg, x, y, cell_size, indices[i, j], color_palette,  color_indices[i, j])
    
    # Draw hierarchical separators for columns
    column_shape = grid_shape[1]
    for r in range(1, rank(column_shape)):
        x_step = cell_size * size(column_shape[:r])
        pos = [(x_step*j + label_margin, label_margin) for j in range(column_shape[r])]
        _draw_hierarchical_separators(dwg, pos, cell_size*M, 'vertical', level=rank(column_shape) - r - 1)


    # Add top axis labels (column indices)
    top_labels = [
        (idx, j * cell_size + label_margin + cell_size // 2, label_margin // 2)
        for j, idx in enumerate(_arange_shape(column_shape).flatten(order='F'))
    ]
    _draw_axis_labels(dwg, top_labels)

    # Draw hierarchical separators for rows
    row_shape = grid_shape[0]
    for r in range(1, rank(row_shape)):
        y_step = cell_size * size(row_shape[:r])
        pos = [(label_margin, y_step*i + label_margin) for i in range(row_shape[r] + 1)]
        _draw_hierarchical_separators(dwg, pos, cell_size*N, 'horizontal', level=rank(row_shape) - r - 1)
    
    # Add left axis labels (row indices)
    left_labels = [
        (idx, label_margin // 2, i * cell_size + label_margin + cell_size // 2)
        for i, idx in enumerate(_arange_shape(row_shape).flatten(order='F'))
    ]    
    _draw_axis_labels(dwg, left_labels)

    return dwg


def _create_coord_space_svg(layout, coord_space=None, color_palette=None):
    """
    Internal helper to create SVG Drawing object for a layout's coordinate space.

    Args:
        layout: CuTe layout object
        coord_space: Shape to use for coordinate space (optional)
                     If None, uses flattened sizes from layout
        color_palette: List of (hue, saturation, lightness) tuples (optional)
                      If None, uses default palette based on rank

    Returns:
        svgwrite.Drawing object
    """
    # Use coord_space if provided, otherwise default to layout's flattened sizes)
    if coord_space is None:
        if isinstance(layout.shape, tuple):
            coord_space = tuple(size(dim) for dim in layout.shape)
        else:
            coord_space = size(layout)
        
    r = rank(coord_space)
    
    # Extract indices
    indices = _extract_layout_indices(layout, coord_space)
    
    # Generate color palette if not provided
    if color_palette is None:
        if cosize(layout) > 16:
            color_palette = create_natural_palette(cosize(layout))
        else: 
            color_palette = create_greyscale_palette(cosize(layout), reverse=True)
    
    # Render grid
    if r == 1:
        return _create_1d_grid_svg(coord_space, indices, color_palette)
    elif r >= 2:
        return _create_2d_grid_svg(coord_space, indices, color_palette)
    else:
        raise ValueError("coord_space must have at least rank 1")


def render_layout(layout, output_file, coord_space=None, color_palette=None):
    """
    Render a CuTe layout's coordinate space as an SVG grid with color-coded cells.

    Args:
        layout: CuTe layout object
        output_file: Output SVG file path
        coord_space: Optional shape to use for coordinate space
        color_palette: Optional list of (hue, saturation, lightness) tuples
    """
    dwg = _create_coord_space_svg(layout, coord_space, color_palette)
    dwg.saveas(output_file)

def display_layout(layout, coord_space=None, indice_space=None, color_palette=None):
    """
    Display a complete layout visualization directly in Jupyter notebooks without writing to disk.
    
    Creates an inline display with two grids side-by-side connected by an arrow:
    - Left: Coordinate space (input coordinates)
    - Right: Indice space (output indices)
    
    Both spaces use the same color palette based on grid position for visual correspondence.
    
    Args:
        layout: CuTe layout object
        coord_space: Optional shape for coordinate space. If None, inferred from layout
        indice_space: Optional shape for indice space. If None, uses layout's codomain
        color_palette: Optional list of (hue, saturation, lightness) tuples
    
    Returns:
        IPython display object
    """
    from IPython.display import SVG, display
    
    dwg = _create_coord_space_svg(layout, coord_space, indice_space, color_palette)
    svg_string = dwg.tostring()
    return display(SVG(svg_string))
