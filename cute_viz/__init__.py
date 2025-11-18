"""
cute_viz: Visualization package for CuTe layouts

This package provides functions to visualize CuTe tensor layouts as SVG images.
"""

from .core import (
    render_layout,
    display_layout,
    create_natural_palette,
    create_greyscale_palette,
)

__version__ = "0.1.0"
__all__ = [
    "render_layout",
    "display_layout",
    "create_natural_palette",
    "create_greyscale_palette",
]