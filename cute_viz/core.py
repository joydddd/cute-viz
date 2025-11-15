"""
Core visualization functions for CuTe layouts.
"""

import colorsys
import itertools
import numpy as np
import svgwrite
from typing import Tuple
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
        
        colors.append((hue, sat, light))
    
    return colors


def _create_greyscale_palette(num_steps, reverse=False):
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


def _create_natural_palette(shape: Tuple[int, ...]):
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
        palette_64 = _create_natural_palette(64)
        # Returns: [(hue0, sat0, light0), (hue0, sat1, light1), ...]  (1D list)
        
        # Tuple specifies (num_steps, num_hues) and returns 2D structure
        full_palette = _create_natural_palette((8, 8))
        # Returns: [[(hue0, sat0, light0), ...], [(hue1, sat0, light0), ...], ...]  (2D list)
    """
    is_2d_color_platte = isinstance(shape, tuple)
    if is_2d_color_platte:
        num_steps, num_hues = shape
    else:
        num_steps = int(np.sqrt(shape))
        num_hues = (shape + num_steps - 1) // num_steps
    
    # Generate saturation/lightness pairs
    sat_light_pairs = _create_color_scale(
        num_steps if num_steps >= 8 else  len(LIGHTNESS_SATURATION_PALETTE_COLOR),
        hue=0,  # Placeholder, will be replaced
        saturation=0.9,  # Max saturation at dark end
        lightness_range=(0.50, 0.90),  # Dark to light range
        vary_saturation=True  # Saturation decreases as lightness increases
    )
    
    if num_hues < len(HUE_PALETTE):
        hue_palette = HUE_PALETTE
    else:
        # Generate evenly distributed hues around color wheel
        hue_palette = [i * 360 / num_hues for i in range(num_hues)]
    
    colors = np.empty((num_hues, num_steps), dtype=object)
    # Multi-hue palette: hue as outer dimension
    # Product: for each hue, iterate through all (sat, light) pairs
    for i, hue in enumerate(hue_palette):
        for j, (_, sat, light) in enumerate(sat_light_pairs):
            colors[i, j] = (hue, sat, light)
    
    if is_2d_color_platte:
        return colors
    else:
        return colors.flatten()


# Combine hue and lightness/saturation palettes to create full HSL color space
# Format: (Hue in degrees 0-360, Saturation 0-1, Lightness 0-1)
# Generate dynamically using natural palette function with hue as outer dimension
HSL_COLORS = _create_natural_palette(
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
    
    # Step 1: Allocate array of shape flatten(coord_space)
    # Use object dtype to store tuples or ints
    flatten_shape = _flatten_to_tuple(coord_space)
    indices = np.empty(flatten_shape, dtype=object)
    
    # Step 2: Iterate over all multi-dimensional coordinates
    for flatten_coord in itertools.product(*[range(d) for d in flatten_shape]):
        coord = _unflatten(flatten_coord, coord_space)
        if isinstance(layout, cute.Tensor):
            indice = _index_tensor(layout, coord)
        else:
            indice = _apply_layout(layout, coord)
        indices[flatten_coord] = indice
    
    return indices


def _extract_layout_coords(layout: cute.Tensor, coord_space: cute.Shape, indice_space: cute.Shape):
    flatten_coord_shape = _flatten_to_tuple(coord_space)
    flatten_indice_shape = _flatten_to_tuple(indice_space)
    coords = np.empty(flatten_indice_shape, dtype=object)
    
    # Step 2: Iterate over all multi-dimensional coordinates
    for flatten_coord in itertools.product(*[range(d) for d in flatten_coord_shape]):
        coord = _unflatten(flatten_coord, coord_space)
        indice = _index_tensor(layout, coord)
        coords[indice] = coord
    
    return coords




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
        hue, sat, light = colors[color_idx % len(colors)]
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
    is_2d_idx = isinstance(idx, (tuple, list))
    if is_2d_idx:
        # 2D layout: display first and second index on separate lines
        i, j = idx
        dwg.add(
            dwg.text(
                str(i),
                insert=(x + cell_size // 2, y + 1 * cell_size // 4),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="8px",
                fill=text_color,
            )
        )
        dwg.add(
            dwg.text(
                str(j),
                insert=(x + cell_size // 2, y + 3 * cell_size // 4),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="8px",
                fill=text_color,
            )
        )
    else:
        # Regular layout: display single centered index
        dwg.add(
            dwg.text(
                str(idx),
                insert=(x + cell_size // 2, y + cell_size // 2),
                text_anchor="middle",
                alignment_baseline="central",
                font_size="8px",
                fill=text_color,
            )
        )


def _draw_axis_labels(dwg, positions, label_margin, cell_size, orientation='horizontal'):
    """
    Draw axis labels at specified positions.
    
    Args:
        dwg: svgwrite.Drawing object
        positions: List of (index, x, y) tuples for label positions
        label_margin: Margin size
        cell_size: Cell size
        orientation: 'horizontal' (top) or 'vertical' (left)
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
    _draw_axis_labels(dwg, label_positions, label_margin, cell_size, 'horizontal')
    
    return dwg


def _create_2d_grid_svg(grid_shape, indices, color_palette, color_indices=None):
    """Create SVG for 2D grid visualization.
    
    Args:
        grid_shape: Shape of the grid to draw (tuple of (M, N))
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
    
    if rank(color_indices.flat[0]) == 1:
        color_palette = color_palette.flatten()
    

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

    # Add top axis labels (column indices)
    top_labels = [
        (j, j * cell_size + label_margin + cell_size // 2, label_margin // 2)
        for j in range(N)
    ]
    _draw_axis_labels(dwg, top_labels, label_margin, cell_size, 'horizontal')
    
    # Add left axis labels (row indices)
    left_labels = [
        (i, label_margin // 2, i * cell_size + label_margin + cell_size // 2)
        for i in range(M)
    ]
    _draw_axis_labels(dwg, left_labels, label_margin, cell_size, 'vertical')

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
    # Use coord_space if provided, otherwise default to layout's flattened sizes
    r = rank(layout)
    if coord_space is None:
        d = depth(layout)
        if d == 0:
            coord_space = size(layout)
        else: 
            coord_space = tuple(size(layout, mode=[i]) for i in range(r))
    
    # Extract indices
    coords = _extract_layout_indices(layout, coord_space)
    
    # Generate color palette if not provided
    if color_palette is None:
        if cosize(layout) > 16:
            color_palette = _create_natural_palette(cosize(layout))
        else: 
            color_palette = _create_greyscale_palette(cosize(layout), reverse=True)
    
    # Render grid
    if r == 1:
        M = coord_space[0] if isinstance(coord_space, tuple) else size(coord_space)
        return _create_1d_grid_svg(M, coords, color_palette)
    elif r >= 2:
        if isinstance(coord_space, tuple):
            M, N = coord_space[0], coord_space[1]
        else:
            M, N = size(coord_space[0]), size(coord_space[1])
        return _create_2d_grid_svg((M, N), coords, color_palette)
    else:
        raise ValueError("coord_space must have at least rank 1")



@cute.jit
def _extract_copy_layout_coords(layout_s, layout_d, num_threads, num_values):
    """
    Extract coordinates from source and destination TV layouts using compile-time loops.

    Args:
        layout_s: CuTe source TV layout object
        layout_d: CuTe destination TV layout object
        num_threads: Number of threads
        num_values: Number of values per thread

    Returns:
        Tuple of two 2D numpy arrays of (i, j) coordinates for each (tid, vid) pair
    """
    coords_s = np.zeros((num_threads, num_values, 2), dtype=np.int32)
    coords_d = np.zeros((num_threads, num_values, 2), dtype=np.int32)

    for tid in range_constexpr(num_threads):
        for vid in range_constexpr(num_values):
            i_s, j_s = layout_s[(tid, vid)]
            coords_s[tid, vid, 0] = i_s
            coords_s[tid, vid, 1] = j_s

            i_d, j_d = layout_d[(tid, vid)]
            coords_d[tid, vid, 0] = i_d
            coords_d[tid, vid, 1] = j_d

    return coords_s, coords_d


def _create_indice_space_svg(layout, indice_space=None, color_palette=None, color_space="coord"):
    """
    Internal helper to create SVG Drawing object for an indice space.

    Args:
        layout: CuTe layout object (rank-1 or rank-2)
        indice_space: Indice space shape (rank-1 or rank-2, optional)
                     If None, uses layout's codomain (cosize)
        color_palette: List of (hue, saturation, lightness) tuples (optional)
                      If None, uses HSL_COLORS
        color_space: Coloring mode - either "coord" or "indices"
                    - "coord": Color cells by source coordinate (coord space position)
                    - "indices": Color cells by target indice (indice space position)
                    Default: "coord"

    Returns:
        svgwrite.Drawing object
    """
    # Validate color_space parameter
    if color_space not in ["coord", "indices"]:
        raise ValueError(f"color_space must be 'coord' or 'indices', got '{color_space}'")
    
    # Determine indice space (default to codomain)
    if indice_space is None:
        indice_space = (cosize(layout),)
    indice_rank = rank(indice_space)
    
    # Compose with identity tensor to map indice_space coords to themselves
    indice_tensor = make_identity_tensor(indice_space)
    composed_layout = cute.composition(indice_tensor, layout)


    if depth(layout) == 0:
        coord_space = size(layout)
    else: 
        coord_space = tuple(size(layout, mode=[i]) for i in range(rank(layout)))
    
    # Generate color palette if not provided
    if color_palette is None:
        if size(layout) > 16:
            color_palette = _create_natural_palette(coord_space)
        else: 
            color_palette = _create_greyscale_palette(coord_space, reverse=True)
        
    coord_array = _extract_layout_coords(composed_layout, coord_space, indice_space)
    
    # Extract forward mapping and create inverse mapping for display
    if indice_rank == 1:
        # 1D indice space
        M = size(indice_space)
        color_idx = coord_array if color_space == "coord" else np.arange(M)
        return _create_1d_grid_svg(M, coord_array, color_palette, color_idx)

    else:
        # 2D indice space
        assert indice_rank == 2, "Expected a rank-2 indice space"
        M, N = size(indice_space[0]), size(indice_space[1])
        color_idx = coord_array if color_space == "coord" else np.arange(M * N).reshape(M, N)
        return _create_2d_grid_svg((M, N), coord_array, color_palette, color_idx)


def render_coord_space_svg(layout, output_file, coord_space=None, color_palette=None):
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


def render_indice_space_svg(layout, output_file, indice_space=None, color_palette=None, color_space="coord"):
    """
    Render a CuTe indice space layout as an SVG grid.

    Args:
        layout: CuTe layout object (rank-1 or rank-2)
        output_file: Output SVG file path
        indice_space: Indice space shape (rank-1 or rank-2, optional)
                     If None, uses layout's codomain
        color_palette: Optional list of (hue, saturation, lightness) tuples
        color_space: Coloring mode - either "coord" or "indices"
                    - "coord": Color cells by source coordinate (default)
                    - "indices": Color cells by target indice
    """
    dwg = _create_indice_space_svg(layout, indice_space, color_palette, color_space)
    dwg.saveas(output_file)


def _create_layout_svg(layout, coord_space=None, indice_space=None, color_palette=None):
    """
    Internal helper to create a complete layout visualization SVG Drawing object.
    
    Creates an SVG with two grids side-by-side connected by an arrow:
    - Left: Coordinate space (input coordinates)
    - Right: Indice space (output indices)
    
    Both spaces use the same color palette based on grid position for visual correspondence.
    
    Args:
        layout: CuTe layout object
        coord_space: Optional shape for coordinate space. If None, inferred from layout
        indice_space: Optional shape for indice space. If None, uses layout's codomain
        color_palette: Optional list of (hue, saturation, lightness) tuples
        
    Returns:
        svgwrite.Drawing object
    """
    # Generate color palette if not provided (same logic as _create_coord_space_svg)
    if indice_space is None:
        indice_space = cosize(layout)
    if color_palette is None:
        if size(indice_space) > 16:
            color_palette = _create_natural_palette(indice_space)
        else:
            color_palette = _create_greyscale_palette(indice_space, reverse=True)
    
    # Create coordinate space grid
    dwg_coord = _create_coord_space_svg(layout, coord_space, color_palette)
    
    # Get dimensions of coord space SVG
    coord_width, coord_height = dwg_coord.attribs['width'], dwg_coord.attribs['height']
    coord_width = float(coord_width) if isinstance(coord_width, str) else coord_width
    coord_height = float(coord_height) if isinstance(coord_height, str) else coord_height
    
    # Create indice space visualization with position-based coloring
    # This makes colors match spatially between coord and indice spaces
    dwg_indice = _create_indice_space_svg(layout, indice_space, color_palette, color_space="indices")
    
    # Get dimensions of indice space SVG
    indice_width, indice_height = dwg_indice.attribs['width'], dwg_indice.attribs['height']
    indice_width = float(indice_width) if isinstance(indice_width, str) else indice_width
    indice_height = float(indice_height) if isinstance(indice_height, str) else indice_height
    
    # Calculate combined dimensions with gap for arrow
    arrow_gap = 60
    title_height = 25  # Space for titles above grids
    total_width = coord_width + arrow_gap + indice_width
    total_height = max(coord_height, indice_height) + title_height
    
    # Create combined SVG
    dwg = svgwrite.Drawing(size=(total_width, total_height))
    
    # Add title for coordinate space
    dwg.add(dwg.text(
        'Coordinate Space',
        insert=(coord_width / 2, 15),
        text_anchor='middle',
        font_size='14px',
        font_weight='bold',
        fill='black'
    ))
    
    # Add coordinate space (left side, shifted down for title)
    coord_group = dwg.g(id='coord_space', transform=f'translate(0, {title_height})')
    for element in dwg_coord.elements:
        coord_group.add(element)
    dwg.add(coord_group)
    
    # Add title for indice space
    offset_x = coord_width + arrow_gap
    dwg.add(dwg.text(
        'Indice Space',
        insert=(offset_x + indice_width / 2, 15),
        text_anchor='middle',
        font_size='14px',
        font_weight='bold',
        fill='black'
    ))
    
    # Add indice space (right side, shifted down for title)
    indice_group = dwg.g(id='indice_space', transform=f'translate({offset_x}, {title_height})')
    for element in dwg_indice.elements:
        indice_group.add(element)
    dwg.add(indice_group)
    
    # Add arrow connecting the two
    arrow_y = total_height / 2
    arrow_start_x = coord_width + 10
    arrow_end_x = coord_width + arrow_gap - 10
    
    # Draw arrow line
    dwg.add(dwg.line(
        start=(arrow_start_x, arrow_y),
        end=(arrow_end_x, arrow_y),
        stroke='black',
        stroke_width=2
    ))
    
    # Draw arrowhead
    arrow_size = 8
    dwg.add(dwg.polygon(
        points=[
            (arrow_end_x, arrow_y),
            (arrow_end_x - arrow_size, arrow_y - arrow_size/2),
            (arrow_end_x - arrow_size, arrow_y + arrow_size/2)
        ],
        fill='black'
    ))
    
    # Add label
    dwg.add(dwg.text(
        'layout',
        insert=(coord_width + arrow_gap/2, arrow_y - 10),
        text_anchor='middle',
        font_size='12px',
        font_style='italic'
    ))
    
    return dwg


def render_layout(layout, output_file, coord_space=None, indice_space=None, color_palette=None):
    """
    Render a complete layout visualization showing both coordinate and indice spaces.
    
    Creates an SVG with two grids side-by-side connected by an arrow:
    - Left: Coordinate space (input coordinates)
    - Right: Indice space (output indices)
    
    Both spaces use the same color palette based on grid position for visual correspondence.
    
    Args:
        layout: CuTe layout object
        output_file: Output SVG file path
        coord_space: Optional shape for coordinate space. If None, inferred from layout
        indice_space: Optional shape for indice space. If None, uses layout's codomain
        color_palette: Optional list of (hue, saturation, lightness) tuples
    """
    dwg = _create_layout_svg(layout, coord_space, indice_space, color_palette)
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
    
    dwg = _create_layout_svg(layout, coord_space, indice_space, color_palette)
    svg_string = dwg.tostring()
    return display(SVG(svg_string))


def display_svg(file_path):
    """
    Display an SVG file in Jupyter notebooks.

    Args:
        file_path: Path to the SVG file to display

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display

    with open(file_path, "r") as f:
        svg_content = f.read()

    return display(SVG(svg_content))


def display_coord_space(layout, coord_space=None, color_palette=None):
    """
    Display a CuTe layout's coordinate space directly in Jupyter notebooks without writing to disk.

    Args:
        layout: CuTe layout object
        coord_space: Optional shape to use for coordinate space
        color_palette: Optional list of (hue, saturation, lightness) tuples

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display
    dwg = _create_coord_space_svg(layout, coord_space, color_palette)
    svg_string = dwg.tostring()
    return display(SVG(svg_string))


def display_indice_space(layout, indice_space=None, color_palette=None, color_space="coord"):
    """
    Display a CuTe indice space layout directly in Jupyter notebooks without writing to disk.

    Args:
        layout: CuTe layout object (rank-1 or rank-2)
        indice_space: Indice space shape (rank-1 or rank-2, optional)
                     If None, uses layout's codomain
        color_palette: Optional list of (hue, saturation, lightness) tuples
        color_space: Coloring mode - either "coord" or "indices"
                    - "coord": Color cells by source coordinate (default)
                    - "indices": Color cells by target indice

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display

    dwg = _create_indice_space_svg(layout, indice_space, color_palette, color_space)
    svg_string = dwg.tostring()
    return display(SVG(svg_string))


def _create_copy_layout_svg(layout_s, layout_d, tile_mn):
    """
    Internal helper to create SVG Drawing object for a Copy layout.

    Copy layouts show source and destination thread-value mappings side-by-side,
    visualizing how data is copied from source to destination memory locations.

    Args:
        layout_s: CuTe source TV layout object
        layout_d: CuTe destination TV layout object
        tile_mn: Rank-2 MN Tile

    Returns:
        svgwrite.Drawing object
    """
    assert rank(layout_s) == 2, "Expected a rank-2 source TV Layout"
    assert rank(layout_d) == 2, "Expected a rank-2 destination TV Layout"
    assert rank(tile_mn) == 2, "Expected a rank-2 MN Tile"

    hsl_colors = HSL_COLORS
    cell_size = 20
    M, N = size(tile_mn[0]), size(tile_mn[1])
    num_threads = size(layout_s, mode=[0])
    num_values = size(layout_s, mode=[1])

    # Extract coordinates using JIT-compiled function with range_constexpr
    coords_s, coords_d = _extract_copy_layout_coords(
        layout_s, layout_d, num_threads, num_values
    )

    # Horizontal gap between source and destination grids
    gap = 1 * cell_size

    # Add margin for axis labels (1 cell on left for S, 1 cell on top, 1 cell on right for D)
    label_margin = cell_size
    total_width = 2 * N * cell_size + gap + 2 * label_margin
    page_height = M * cell_size + label_margin

    filled_s = np.zeros((M, N), dtype=bool)
    filled_d = np.zeros((M, N), dtype=bool)
    dwg = svgwrite.Drawing(size=(total_width, page_height))

    # Draw source grid (left side) - background cells
    for i in range(M):
        for j in range(N):
            dwg.add(
                dwg.rect(
                    insert=(label_margin + j * cell_size, label_margin + i * cell_size),
                    size=(cell_size, cell_size),
                    fill="white",
                    stroke="black",
                )
            )

    # Draw destination grid (right side) - background cells
    x_offset = label_margin + N * cell_size + gap
    for i in range(M):
        for j in range(N):
            dwg.add(
                dwg.rect(
                    insert=(x_offset + j * cell_size, label_margin + i * cell_size),
                    size=(cell_size, cell_size),
                    fill="white",
                    stroke="black",
                )
            )

    # Fill source grid with thread-value data
    for tid in range(num_threads):
        for vid in range(num_values):
            i, j = int(coords_s[tid, vid, 0]), int(coords_s[tid, vid, 1])
            x = label_margin + j * cell_size
            y = label_margin + i * cell_size

            if filled_s[i, j]:
                continue
            filled_s[i, j] = True
            
            _draw_layout_cell(dwg, x, y, cell_size, (tid, vid), hsl_colors)

    # Fill destination grid with thread-value data
    for tid in range(num_threads):
        for vid in range(num_values):
            i, j = int(coords_d[tid, vid, 0]), int(coords_d[tid, vid, 1])
            x = x_offset + j * cell_size
            y = label_margin + i * cell_size

            if filled_d[i, j]:
                continue
            filled_d[i, j] = True
            
            _draw_layout_cell(dwg, x, y, cell_size, (tid, vid), hsl_colors)

    # Add axis labels for source grid (matching C++ print_latex_copy behavior)
    # Top labels: column indices (0 to N-1)
    top_labels_src = [
        (j, j * cell_size + label_margin + cell_size // 2, label_margin // 2)
        for j in range(N)
    ]
    _draw_axis_labels(dwg, top_labels_src, label_margin, cell_size, 'horizontal')

    # Left labels: row indices (0 to M-1)
    left_labels_src = [
        (i, label_margin // 2, i * cell_size + label_margin + cell_size // 2)
        for i in range(M)
    ]
    _draw_axis_labels(dwg, left_labels_src, label_margin, cell_size, 'vertical')

    # Add axis labels for destination grid
    # Top labels: column indices (0 to N-1)
    top_labels_dst = [
        (j, x_offset + j * cell_size + cell_size // 2, label_margin // 2)
        for j in range(N)
    ]
    _draw_axis_labels(dwg, top_labels_dst, label_margin, cell_size, 'horizontal')

    # Right labels: row indices (0 to M-1) - placed on RIGHT side for D grid
    right_labels_dst = [
        (i, x_offset + N * cell_size + label_margin // 2, i * cell_size + label_margin + cell_size // 2)
        for i in range(M)
    ]
    _draw_axis_labels(dwg, right_labels_dst, label_margin, cell_size, 'vertical')

    return dwg


def render_copy_layout_svg(layout_s, layout_d, tile_mn, output_file):
    """
    Render a CuTe copy layout as an SVG with source and destination grids side-by-side.

    Copy layouts show how threads map to source and destination memory locations,
    visualizing the data movement pattern.

    Args:
        layout_s: CuTe source TV layout object
        layout_d: CuTe destination TV layout object
        tile_mn: Rank-2 MN Tile
        output_file: Output SVG file path
    """
    dwg = _create_copy_layout_svg(layout_s, layout_d, tile_mn)
    dwg.saveas(output_file)


def display_copy_layout(layout_s, layout_d, tile_mn):
    """
    Display a CuTe copy layout directly in Jupyter notebooks without writing to disk.

    Args:
        layout_s: CuTe source TV layout object
        layout_d: CuTe destination TV layout object
        tile_mn: Rank-2 MN Tile

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display

    dwg = _create_copy_layout_svg(layout_s, layout_d, tile_mn)
    svg_string = dwg.tostring()
    return display(SVG(svg_string))


###################################
# TiledCopy Layout Extraction Utilities
###################################


def tidfrg_S(tiled_copy, tile_mn):
    """
    Extract source thread-value layout from TiledCopy.

    Python equivalent of C++ TiledCopy::tidfrg_S().

    This function creates a tensor that maps (thread_id, value_id) coordinates
    to (m, n) spatial coordinates for the source layout of a copy operation.

    Args:
        tiled_copy: CuTe TiledCopy object
        tile_mn: Tile shape as (M, N) tuple or Shape object

    Returns:
        Tensor mapping (thread_id, value_id) -> (m, n) coordinates for source

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import tidfrg_S
        >>>
        >>> copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        >>> thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
        >>> val_layout = cute.make_ordered_layout((2, 1), order=(1, 0))
        >>> tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        >>>
        >>> layout_s = tidfrg_S(tiled_copy, (8, 8))
    """
    ref_tensor = make_identity_tensor(tile_mn)
    return cute.composition(ref_tensor, tiled_copy.layout_src_tv_tiled)


def tidfrg_D(tiled_copy, tile_mn):
    """
    Extract destination thread-value layout from TiledCopy.

    Python equivalent of C++ TiledCopy::tidfrg_D().

    This function creates a tensor that maps (thread_id, value_id) coordinates
    to (m, n) spatial coordinates for the destination layout of a copy operation.

    Args:
        tiled_copy: CuTe TiledCopy object
        tile_mn: Tile shape as (M, N) tuple or Shape object

    Returns:
        Tensor mapping (thread_id, value_id) -> (m, n) coordinates for destination

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import tidfrg_D
        >>>
        >>> copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        >>> thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
        >>> val_layout = cute.make_ordered_layout((2, 1), order=(1, 0))
        >>> tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        >>>
        >>> layout_d = tidfrg_D(tiled_copy, (8, 8))
    """
    ref_tensor = make_identity_tensor(tile_mn)
    return cute.composition(ref_tensor, tiled_copy.layout_dst_tv_tiled)


###################################
# High-Level TiledCopy Visualization API
###################################


def render_tiled_copy_svg(tiled_copy, tile_mn, output_path):
    """
    Render a TiledCopy visualization to SVG file.

    Python equivalent of C++ print_latex(TiledCopy).

    This high-level function automatically extracts source and destination
    thread-value layouts from a TiledCopy object and renders them side-by-side.

    Args:
        tiled_copy: CuTe TiledCopy object created with make_tiled_copy_tv()
        tile_mn: Tile shape as (M, N) tuple
        output_path: Path to save the SVG file

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import render_tiled_copy_svg
        >>>
        >>> # Create copy atom and tiled copy
        >>> copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        >>> thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
        >>> val_layout = cute.make_ordered_layout((2, 1), order=(1, 0))
        >>> tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        >>>
        >>> # Render in one call!
        >>> render_tiled_copy_svg(tiled_copy, (8, 8), "copy_layout.svg")
    """
    # Extract source and destination TV layouts
    tensor_s_tv = tidfrg_S(tiled_copy, tile_mn)
    tensor_d_tv = tidfrg_D(tiled_copy, tile_mn)

    # Handle potential extra dimensions (like C++ (_,_,Int<0>{}))
    # If tensors have more than 2 dimensions, slice to get the first element
    if hasattr(tensor_s_tv, 'ndim') and tensor_s_tv.ndim > 2:
        tensor_s_tv = tensor_s_tv[:, :, 0]
    if hasattr(tensor_d_tv, 'ndim') and tensor_d_tv.ndim > 2:
        tensor_d_tv = tensor_d_tv[:, :, 0]

    # Render the copy layout
    render_copy_layout_svg(tensor_s_tv, tensor_d_tv, tile_mn, output_path)


def display_tiled_copy(tiled_copy, tile_mn):
    """
    Display a TiledCopy visualization in Jupyter notebook.

    Python equivalent of C++ print_latex(TiledCopy) for interactive notebooks.

    Args:
        tiled_copy: CuTe TiledCopy object created with make_tiled_copy_tv()
        tile_mn: Tile shape as (M, N) tuple

    Returns:
        IPython.display.SVG object for inline display

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import display_tiled_copy
        >>>
        >>> # Create copy atom and tiled copy
        >>> copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        >>> thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
        >>> val_layout = cute.make_ordered_layout((2, 1), order=(1, 0))
        >>> tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        >>>
        >>> # Display inline in Jupyter
        >>> display_tiled_copy(tiled_copy, (8, 8))
    """
    # Extract source and destination TV layouts
    tensor_s_tv = tidfrg_S(tiled_copy, tile_mn)
    tensor_d_tv = tidfrg_D(tiled_copy, tile_mn)

    # Handle potential extra dimensions
    if hasattr(tensor_s_tv, 'ndim') and tensor_s_tv.ndim > 2:
        tensor_s_tv = tensor_s_tv[:, :, 0]
    if hasattr(tensor_d_tv, 'ndim') and tensor_d_tv.ndim > 2:
        tensor_d_tv = tensor_d_tv[:, :, 0]

    # Display the copy layout
    return display_copy_layout(tensor_s_tv, tensor_d_tv, tile_mn)


###################################
# MMA (Matrix Multiply-Accumulate) Visualization
###################################


@cute.jit
def _extract_mma_coords(tensorC, tensorA, tensorB, num_threads_C, num_values_C, num_threads_A, num_values_A, num_threads_B, num_values_B):
    """
    Extract coordinates from A, B, and C tensors using compile-time loops.

    Args:
        tensorC: C matrix TV tensor
        tensorA: A matrix TV tensor
        tensorB: B matrix TV tensor
        num_threads_C, num_values_C: Thread and value counts for C
        num_threads_A, num_values_A: Thread and value counts for A
        num_threads_B, num_values_B: Thread and value counts for B

    Returns:
        Tuple of three arrays with coordinates for C, A, B
    """
    coords_C = np.zeros((num_threads_C, num_values_C, 2), dtype=np.int32)
    coords_A = np.zeros((num_threads_A, num_values_A, 2), dtype=np.int32)
    coords_B = np.zeros((num_threads_B, num_values_B, 2), dtype=np.int32)

    for tid in range_constexpr(num_threads_C):
        for vid in range_constexpr(num_values_C):
            m, n = tensorC[tid, vid]
            coords_C[tid, vid, 0] = m
            coords_C[tid, vid, 1] = n

    for tid in range_constexpr(num_threads_A):
        for vid in range_constexpr(num_values_A):
            m, k = tensorA[tid, vid]
            coords_A[tid, vid, 0] = m
            coords_A[tid, vid, 1] = k

    for tid in range_constexpr(num_threads_B):
        for vid in range_constexpr(num_values_B):
            n, k = tensorB[tid, vid]
            coords_B[tid, vid, 0] = n
            coords_B[tid, vid, 1] = k

    return coords_C, coords_A, coords_B


def _create_mma_layout_svg(tiled_mma, tile_mnk):
    """
    Create SVG visualization of MMA layout showing A, B, and C matrices.

    Layout:
        B
    A   C

    Where C = A × B for matrix multiplication.

    Args:
        tiled_mma: TiledMMA object
        tile_mnk: Tuple of (M, N, K) tile dimensions

    Returns:
        svgwrite.Drawing object
    """
    M, N, K = tile_mnk

    # Extract TV layouts from TiledMMA
    layoutC_TV = tiled_mma.tv_layout_C_tiled
    layoutA_TV = tiled_mma.tv_layout_A_tiled
    layoutB_TV = tiled_mma.tv_layout_B_tiled

    # Create identity tensors and compose
    refC = make_identity_tensor((M, N))
    tensorC_TV = cute.composition(refC, layoutC_TV)

    refA = make_identity_tensor((M, K))
    tensorA_TV = cute.composition(refA, layoutA_TV)

    refB = make_identity_tensor((N, K))
    tensorB_TV = cute.composition(refB, layoutB_TV)

    # Handle potential extra dimensions
    tensorC = tensorC_TV[:, :, 0] if hasattr(tensorC_TV, 'ndim') and tensorC_TV.ndim > 2 else tensorC_TV
    tensorA = tensorA_TV[:, :, 0] if hasattr(tensorA_TV, 'ndim') and tensorA_TV.ndim > 2 else tensorA_TV
    tensorB = tensorB_TV[:, :, 0] if hasattr(tensorB_TV, 'ndim') and tensorB_TV.ndim > 2 else tensorB_TV

    cell_size = 20

    # Add margin for axis labels
    label_margin = 0

    # SVG dimensions
    page_width = (K + N + 2) * cell_size + label_margin
    page_height = (K + M + 2) * cell_size + label_margin

    dwg = svgwrite.Drawing(size=(page_width, page_height))

    # Track filled cells to avoid duplicates
    import numpy as np
    filled = np.zeros((M, N, K), dtype=bool)

    # Get number of threads and values for each tensor
    num_threads_C = size(tensorC, mode=[0])
    num_values_C = size(tensorC, mode=[1])
    num_threads_A = size(tensorA, mode=[0])
    num_values_A = size(tensorA, mode=[1])
    num_threads_B = size(tensorB, mode=[0])
    num_values_B = size(tensorB, mode=[1])

    # Extract coordinates from tensors (this happens in JIT context)
    coords_C, coords_A, coords_B = _extract_mma_coords(
        tensorC, tensorA, tensorB,
        num_threads_C, num_values_C,
        num_threads_A, num_values_A,
        num_threads_B, num_values_B
    )

    # Colors (matching C++ SVGColor_TV from print_svg.hpp)
    hsl_colors = HSL_COLORS

    # --- Draw C (M×N at bottom-right) ---
    for tid in range(num_threads_C):
        for vid in range(num_values_C):
            m, n = int(coords_C[tid, vid, 0]), int(coords_C[tid, vid, 1])
            if m < M and n < N and not filled[m, n, 0]:
                filled[m, n, 0] = True

                x = label_margin + (n + K + 2) * cell_size
                y = label_margin + (m + K + 2) * cell_size

                _draw_layout_cell(dwg, x, y, cell_size, (tid, vid), hsl_colors)

    # Reset filled tracker
    filled.fill(False)

    # --- Draw A (M×K at left) ---
    for tid in range(num_threads_A):
        for vid in range(num_values_A):
            m, k = int(coords_A[tid, vid, 0]), int(coords_A[tid, vid, 1])
            if m < M and k < K and not filled[m, 0, k]:
                filled[m, 0, k] = True

                x = label_margin + (k + 1) * cell_size
                y = label_margin + (m + K + 2) * cell_size

                # Convert HSL to RGB (0-255 range)
                hue, sat, light = hsl_colors[tid % len(hsl_colors)]
                rgb = tuple(int(c * 255) for c in colorsys.hls_to_rgb(hue / 360.0, light, sat))

                rect = dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(*rgb, mode="RGB"),
                    stroke='black'
                )
                dwg.add(rect)

                # Thread ID
                text1 = dwg.text(
                    f'T{tid}',
                    insert=(x + cell_size/2, y + cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text1)

                # Value ID
                text2 = dwg.text(
                    f'V{vid}',
                    insert=(x + cell_size/2, y + 3*cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text2)

    # Reset filled tracker
    filled.fill(False)

    # --- Draw B (N×K at top, shown as K×N transposed) ---
    for tid in range(num_threads_B):
        for vid in range(num_values_B):
            n, k = int(coords_B[tid, vid, 0]), int(coords_B[tid, vid, 1])
            if n < N and k < K and not filled[0, n, k]:
                filled[0, n, k] = True

                x = label_margin + (n + K + 2) * cell_size
                y = label_margin + (k + 1) * cell_size

                # Convert HSL to RGB (0-255 range)
                hue, sat, light = hsl_colors[tid % len(hsl_colors)]
                rgb = tuple(int(c * 255) for c in colorsys.hls_to_rgb(hue / 360.0, light, sat))

                rect = dwg.rect(
                    insert=(x, y),
                    size=(cell_size, cell_size),
                    fill=svgwrite.rgb(*rgb, mode="RGB"),
                    stroke='black'
                )
                dwg.add(rect)

                # Thread ID
                text1 = dwg.text(
                    f'T{tid}',
                    insert=(x + cell_size/2, y + cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text1)

                # Value ID
                text2 = dwg.text(
                    f'V{vid}',
                    insert=(x + cell_size/2, y + 3*cell_size/4),
                    text_anchor='middle',
                    alignment_baseline='central',
                    font_size='8px'
                )
                dwg.add(text2)

    # Add axis labels (matching C++ print_latex_mma behavior)

    # --- A matrix (M×K) axis labels ---
    # Top labels: K dimension (0 to K-1)
    a_top_labels = [
        (k, label_margin + (k + 1) * cell_size + cell_size // 2, label_margin + (K + 2) * cell_size - cell_size // 2)
        for k in range(K)
    ]
    _draw_axis_labels(dwg, a_top_labels, label_margin, cell_size, 'horizontal')

    # Left labels: M dimension (0 to M-1)
    a_left_labels = [
        (m, label_margin + cell_size // 2, label_margin + (m + K + 2) * cell_size + cell_size // 2)
        for m in range(M)
    ]
    _draw_axis_labels(dwg, a_left_labels, label_margin, cell_size, 'vertical')

    # --- B matrix (K×N, shown transposed) axis labels ---
    # Top labels: K dimension (0 to K-1)
    b_left_labels = [
        (k, label_margin + (K + 2) * cell_size - cell_size // 2, label_margin + (k + 1) * cell_size + cell_size // 2)
        for k in range(K)
    ]
    _draw_axis_labels(dwg, b_left_labels, label_margin, cell_size, 'vertical')

    # Right labels: N dimension (0 to N-1)
    b_top_labels = [
        (n, label_margin + (n + K + 2) * cell_size + cell_size // 2, label_margin + cell_size // 2)
        for n in range(N)
    ]
    _draw_axis_labels(dwg, b_top_labels, label_margin, cell_size, 'horizontal')

    return dwg


def render_mma_layout_svg(tiled_mma, tile_mnk, output_file):
    """
    Render a TiledMMA layout as an SVG showing A, B, and C matrix thread mappings.

    Alias for render_tiled_mma_svg().

    Args:
        tiled_mma: CuTe TiledMMA object
        tile_mnk: Tuple (M, N, K) tile dimensions
        output_file: Output SVG file path
    """
    dwg = _create_mma_layout_svg(tiled_mma, tile_mnk)
    dwg.saveas(output_file)


def render_mma_from_layouts(layoutC, layoutA, layoutB, tile_mnk, output_file):
    """
    Render MMA layout from manually constructed TV layouts.

    Low-level API for custom layout visualization.

    Args:
        layoutC: C matrix TV layout (M×N)
        layoutA: A matrix TV layout (M×K)
        layoutB: B matrix TV layout (N×K)
        tile_mnk: Tuple (M, N, K) tile dimensions
        output_file: Output SVG file path
    """
    M, N, K = tile_mnk

    # Create identity tensors and compose
    refC = make_identity_tensor((M, N))
    tensorC_TV = cute.composition(refC, layoutC)

    refA = make_identity_tensor((M, K))
    tensorA_TV = cute.composition(refA, layoutA)

    refB = make_identity_tensor((N, K))
    tensorB_TV = cute.composition(refB, layoutB)

    # Handle potential extra dimensions
    tensorC = tensorC_TV[:, :, 0] if hasattr(tensorC_TV, 'ndim') and tensorC_TV.ndim > 2 else tensorC_TV
    tensorA = tensorA_TV[:, :, 0] if hasattr(tensorA_TV, 'ndim') and tensorA_TV.ndim > 2 else tensorA_TV
    tensorB = tensorB_TV[:, :, 0] if hasattr(tensorB_TV, 'ndim') and tensorB_TV.ndim > 2 else tensorB_TV

    # Create SVG using the same internal logic
    cell_size = 20
    page_width = (K + N + 2) * cell_size
    page_height = (K + M + 2) * cell_size

    dwg = svgwrite.Drawing(size=(page_width, page_height))

    # Track filled cells
    import numpy as np
    filled = np.zeros((M, N, K), dtype=bool)

    # Get sizes
    num_threads_C = size(tensorC, mode=[0])
    num_values_C = size(tensorC, mode=[1])
    num_threads_A = size(tensorA, mode=[0])
    num_values_A = size(tensorA, mode=[1])
    num_threads_B = size(tensorB, mode=[0])
    num_values_B = size(tensorB, mode=[1])

    # Extract coordinates
    coords_C, coords_A, coords_B = _extract_mma_coords(
        tensorC, tensorA, tensorB,
        num_threads_C, num_values_C,
        num_threads_A, num_values_A,
        num_threads_B, num_values_B
    )

    # Colors (matching C++ SVGColor_TV from print_svg.hpp)
    hsl_colors = HSL_COLORS

    # Draw C
    for tid in range(num_threads_C):
        for vid in range(num_values_C):
            m, n = int(coords_C[tid, vid, 0]), int(coords_C[tid, vid, 1])
            if m < M and n < N and not filled[m, n, 0]:
                filled[m, n, 0] = True
                x = (n + K + 2) * cell_size
                y = (m + K + 2) * cell_size
                # Convert HSL to RGB (0-255 range)
                hue, sat, light = hsl_colors[tid % len(hsl_colors)]
                rgb = tuple(int(c * 255) for c in colorsys.hls_to_rgb(hue / 360.0, light, sat))
                rect = dwg.rect(insert=(x, y), size=(cell_size, cell_size),
                               fill=svgwrite.rgb(*rgb, mode="RGB"), stroke='black')
                dwg.add(rect)
                text1 = dwg.text(f'T{tid}', insert=(x + cell_size/2, y + cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text1)
                text2 = dwg.text(f'V{vid}', insert=(x + cell_size/2, y + 3*cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text2)

    filled.fill(False)

    # Draw A
    for tid in range(num_threads_A):
        for vid in range(num_values_A):
            m, k = int(coords_A[tid, vid, 0]), int(coords_A[tid, vid, 1])
            if m < M and k < K and not filled[m, 0, k]:
                filled[m, 0, k] = True
                x = (k + 1) * cell_size
                y = (m + K + 2) * cell_size
                # Convert HSL to RGB (0-255 range)
                hue, sat, light = hsl_colors[tid % len(hsl_colors)]
                rgb = tuple(int(c * 255) for c in colorsys.hls_to_rgb(hue / 360.0, light, sat))
                rect = dwg.rect(insert=(x, y), size=(cell_size, cell_size),
                               fill=svgwrite.rgb(*rgb, mode="RGB"), stroke='black')
                dwg.add(rect)
                text1 = dwg.text(f'T{tid}', insert=(x + cell_size/2, y + cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text1)
                text2 = dwg.text(f'V{vid}', insert=(x + cell_size/2, y + 3*cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text2)

    filled.fill(False)

    # Draw B
    for tid in range(num_threads_B):
        for vid in range(num_values_B):
            n, k = int(coords_B[tid, vid, 0]), int(coords_B[tid, vid, 1])
            if n < N and k < K and not filled[0, n, k]:
                filled[0, n, k] = True
                x = (n + K + 2) * cell_size
                y = (k + 1) * cell_size
                # Convert HSL to RGB (0-255 range)
                hue, sat, light = hsl_colors[tid % len(hsl_colors)]
                rgb = tuple(int(c * 255) for c in colorsys.hls_to_rgb(hue / 360.0, light, sat))
                rect = dwg.rect(insert=(x, y), size=(cell_size, cell_size),
                               fill=svgwrite.rgb(*rgb, mode="RGB"), stroke='black')
                dwg.add(rect)
                text1 = dwg.text(f'T{tid}', insert=(x + cell_size/2, y + cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text1)
                text2 = dwg.text(f'V{vid}', insert=(x + cell_size/2, y + 3*cell_size/4),
                                text_anchor='middle', alignment_baseline='central', font_size='8px')
                dwg.add(text2)

    dwg.saveas(output_file)


def display_mma_layout(tiled_mma, tile_mnk):
    """
    Display a TiledMMA layout directly in Jupyter notebooks.

    Alias for display_tiled_mma().

    Args:
        tiled_mma: CuTe TiledMMA object
        tile_mnk: Tuple (M, N, K) tile dimensions

    Returns:
        IPython display object
    """
    from IPython.display import SVG, display
    dwg = _create_mma_layout_svg(tiled_mma, tile_mnk)
    return display(SVG(dwg.tostring()))


###################################
# High-Level TiledMMA Visualization API
###################################


def render_tiled_mma_svg(tiled_mma, tile_mnk, output_path):
    """
    Render a TiledMMA visualization to SVG file.

    Python equivalent of C++ print_latex(TiledMMA).

    This high-level function automatically extracts A, B, and C thread-value
    layouts from a TiledMMA object and renders them in the standard layout:
        B
    A   C

    Args:
        tiled_mma: CuTe TiledMMA object created with make_tiled_mma()
        tile_mnk: Tile shape as (M, N, K) tuple
        output_path: Path to save the SVG file

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import render_tiled_mma_svg
        >>>
        >>> # Create MMA operation and tiled MMA
        >>> op = cute.nvgpu.MmaUniversalOp(Float32)
        >>> atoms_layout = cute.make_layout((16, 1, 1), stride=(1, 0, 0))
        >>> tiled_mma = cute.make_tiled_mma(op, atoms_layout)
        >>>
        >>> # Render in one call!
        >>> render_tiled_mma_svg(tiled_mma, (8, 8, 8), "mma_layout.svg")
    """
    dwg = _create_mma_layout_svg(tiled_mma, tile_mnk)
    dwg.saveas(output_path)


def display_tiled_mma(tiled_mma, tile_mnk):
    """
    Display a TiledMMA visualization in Jupyter notebook.

    Python equivalent of C++ print_latex(TiledMMA) for interactive notebooks.

    Args:
        tiled_mma: CuTe TiledMMA object created with make_tiled_mma()
        tile_mnk: Tile shape as (M, N, K) tuple

    Returns:
        IPython.display.SVG object for inline display

    Example:
        >>> from cutlass import cute, Float32
        >>> from cute_viz import display_tiled_mma
        >>>
        >>> # Create MMA operation and tiled MMA
        >>> op = cute.nvgpu.MmaUniversalOp(Float32)
        >>> atoms_layout = cute.make_layout((16, 1, 1), stride=(1, 0, 0))
        >>> tiled_mma = cute.make_tiled_mma(op, atoms_layout)
        >>>
        >>> # Display inline in Jupyter
        >>> display_tiled_mma(tiled_mma, (8, 8, 8))
    """
    from IPython.display import SVG, display
    dwg = _create_mma_layout_svg(tiled_mma, tile_mnk)
    return display(SVG(dwg.tostring()))