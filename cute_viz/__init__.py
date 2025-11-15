"""
cute_viz: Visualization package for CuTe layouts

This package provides functions to visualize CuTe tensor layouts as SVG images.
"""

from .core import (
    render_layout,
    render_coord_space_svg,
    render_indice_space_svg,
    render_copy_layout_svg,
    render_tiled_copy_svg,
    render_mma_layout_svg,
    render_mma_from_layouts,
    render_tiled_mma_svg,
    display_svg,
    display_coord_space,
    display_indice_space,
    display_copy_layout,
    display_tiled_copy,
    display_mma_layout,
    display_tiled_mma,
    tidfrg_S,
    tidfrg_D,
)

__version__ = "0.1.0"
__all__ = [
    "render_layout",
    "render_coord_space_svg",
    "render_indice_space_svg",
    "render_copy_layout_svg",
    "render_tiled_copy_svg",
    "render_mma_layout_svg",
    "render_mma_from_layouts",
    "render_tiled_mma_svg",
    "display_svg",
    "display_coord_space",
    "display_indice_space",
    "display_copy_layout",
    "display_tiled_copy",
    "display_mma_layout",
    "display_tiled_mma",
    "tidfrg_S",
    "tidfrg_D",
]