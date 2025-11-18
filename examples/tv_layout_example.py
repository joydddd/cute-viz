"""
Thread-Value Layout Visualization Example

This example demonstrates how to visualize a CuTe thread-value (TV) layout.
"""

from cutlass import cute
from cute_viz import render_coord_space_svg, create_natural_palette


@cute.jit
def main():
    # Create a thread-value layout
    tile_mn, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_ordered_layout((2, 8), order=(1, 0)),
        val_layout=cute.make_ordered_layout((3, 2), order=(1, 0))
    )
    print("tile_mn: ", tile_mn, "tv_layout: ", tv_layout)

    # Create a tensor following data tensor layout for visualization. 
    identity_tensor = cute.make_identity_tensor((16, 6))
    color_palette = create_natural_palette((16, 6)) # Color the visualization following data tensor layout. 
    transpose = cute.make_ordered_layout((6, 16), order=(1, 0))
    identity_tensor_row_major = cute.composition(identity_tensor, transpose)


    # Coord space is ((tid_x, tid_y), (vid_x, vid_y)), where x dimension is continuous. 
    coord_space = ((8, 2), (2, 3))


    # Render to SVG file
    render_coord_space_svg(cute.composition(identity_tensor_row_major, tv_layout), "assets/tv_layout.svg", coord_space=coord_space, color_palette=color_palette)
    # render_indice_space_svg(tv_layout, "assets/tv_layout.svg", tile_mn)
    print("TV layout saved to assets/tv_layout.svg")

    # Or display directly in Jupyter notebook
    # Uncomment the line below when running in Jupyter
    # display_indice_space(tv_layout, tile_mn)


if __name__ == "__main__":
    main()
