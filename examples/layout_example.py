"""
Basic Layout Visualization Example

This example demonstrates how to visualize a CuTe layout.
"""

from cutlass import cute
from cute_viz import render_layout, display_coord_space


@cute.jit
def main():
    # Create a layout with shape (8, 8) and stride (8, 1)
    # This is a row-major layout where indices increase sequentially
    layout = cute.make_layout((8, 8), stride=(8, 1))

    # Render to SVG file
    render_layout(layout, "assets/layout.svg")
    print("Layout saved to assets/layout.svg")

    # Or display directly in Jupyter notebook
    # Uncomment the line below when running in Jupyter
    # display_coord_space(layout)


    # Example 3: Vector layout
    # Simple contiguous vector of size 10
    layout_vector = cute.make_layout(10)
    print(f"Vector Layout (10:1): {layout_vector}")
    render_layout(layout_vector, "assets/layout_vector.svg")
    print("Vector layout saved to assets/layout_vector.svg")
    
    print()
    
    # Example 4: Matrix with custom stride
    # Column-major with gaps: (8, 8):(1, 10)
    # Stride of 1 down columns, stride of 10 across rows (creating gaps)
    layout_custom = cute.make_layout((8, 8), stride=(1, 10))
    print(f"Custom stride layout (8,8):(1,10): {layout_custom}")
    render_layout(layout_custom, "assets/layout_custom_stride.svg")
    print("Custom stride layout saved to assets/layout_custom_stride.svg")
    
    print()
    
    # Example 5: Hierarchical layout
    # First mode is nested (2, 3), second mode is 4
    # Strides: (1, 2) for nested mode, 6 for outer mode
    layout_hierarchical = cute.make_layout(((2, 3), 4), stride=((1, 8), 2))
    print(f"Hierarchical layout ((2,3),4):((1,2),6): {layout_hierarchical}")
    render_layout(layout_hierarchical, "assets/layout_hierarchical.svg")
    print("Hierarchical layout saved to assets/layout_hierarchical.svg")


if __name__ == "__main__":
    main()
