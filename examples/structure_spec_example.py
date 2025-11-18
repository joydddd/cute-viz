"""
Example demonstrating the general structure specification feature for coordinate_structure.

This example shows how to use the coordinate_structure parameter
to control layout visualization with structure specifications that support
ANY hierarchical nesting pattern.

The structure specification mirrors the layout's hierarchical structure:
- 0 means flatten that mode
- Nested tuples preserve hierarchical boundaries
- The specification can match any arbitrarily nested layout structure

Examples:
  Layout ((2, 2), 2):
    - 0: Flatten to 1D (8 elements)
    - (0, 0): Flatten to 2D (4×2 grid)
    - False or ((0,0), 0): Show hierarchical with tile boundaries
  
  Layout ((2,2), (3,4)):
    - True: Flatten completely
    - False: Show both modes hierarchical
    - ((0,0), 0): First mode hierarchical, second flat
    - (0, (0,0)): First mode flat, second hierarchical
"""

from cutlass import cute, range_constexpr
from cute_viz import render_layout_svg, display_layout


@cute.jit
def demonstrate_structure_specs():
    """Demonstrate various structure specification options."""
    
    print("=" * 60)
    print("EXAMPLE: Hierarchical layout ((2, 2), 2)")
    print("=" * 60)
    layout = cute.make_layout(((2, 2), 2))
    print(f"Layout: {layout}")
    print(f"Rank: {cute.rank(layout)}")
    print(f"Size: {cute.size(layout)}")
    print(f"Depth mode 0: {cute.depth(layout[0])}")
    print(f"Depth mode 1: {cute.depth(layout[1])}")
    
    print("\nHierarchical structure: 2x2 elements, 2 tiles")
    print("\nMapping:")
    for i in range_constexpr(2):
        for j in range_constexpr(2):
            for k in range_constexpr(2):
                idx = layout(((i, j), k))
                print(f"(({i},{j}),{k})->{idx}", end="  ")
        print()
    
    # 1D visualization - flatten everything to a single row
    print("\n--- 1D Visualization (coordinate_structure=0) ---")
    render_layout_svg(layout, "assets/structure_spec_1d.svg", coordinate_structure=0)
    print("✓ Saved to assets/structure_spec_1d.svg")
    print("Result: 8 elements in a single horizontal row")
    
    # Alternative 1D syntax
    render_layout_svg(layout, "assets/structure_spec_1d_tuple.svg", coordinate_structure=(0,))
    print("✓ Saved to assets/structure_spec_1d_tuple.svg (same as above)")
    
    # Hierarchical visualization - show tile boundaries (default)
    print("\n--- Hierarchical Visualization (coordinate_structure=None, default) ---")
    render_layout_svg(layout, "assets/structure_spec_hierarchical.svg")
    print("✓ Saved to assets/structure_spec_hierarchical.svg")
    print("Result: 2D grid with blue tile boundaries showing hierarchical structure")
    
    # Explicit hierarchical specification (same as None for this layout)
    render_layout_svg(layout, "assets/structure_spec_hierarchical_explicit.svg", coordinate_structure=((0, 0), 0))
    print("✓ Saved to assets/structure_spec_hierarchical_explicit.svg (same as above)")
    
    # 2D visualization - flatten to a 2D grid
    print("\n--- 2D Visualization (coordinate_structure=(0, 0)) ---")
    render_layout_svg(layout, "assets/structure_spec_2d.svg", coordinate_structure=(0, 0))
    print("✓ Saved to assets/structure_spec_2d.svg")
    print("Result: 4x2 grid (flattened hierarchical structure)")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Another hierarchical layout (2, (2, 2))")
    print("=" * 60)
    layout2 = cute.make_layout((2, (2, 2)), stride=(4, (2, 1)))
    print(f"Layout: {layout2}")
    
    # 1D visualization
    render_layout_svg(layout2, "assets/structure_spec2_1d.svg", coordinate_structure=0)
    print("✓ Saved 1D to assets/structure_spec2_1d.svg")
    
    # Hierarchical (default)
    render_layout_svg(layout2, "assets/structure_spec2_hierarchical.svg")
    print("✓ Saved hierarchical to assets/structure_spec2_hierarchical.svg")
    
    # 2D flattened
    render_layout_svg(layout2, "assets/structure_spec2_2d.svg", coordinate_structure=(0, 0))
    print("✓ Saved 2D flat to assets/structure_spec2_2d.svg")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Complex hierarchical layout ((2,2), (3,4))")
    print("=" * 60)
    layout3 = cute.make_layout(((2, 2), (3, 4)), stride=((1, 6), (2, 12)))
    print(f"Layout: {layout3}")
    print(f"Both modes are hierarchical: (2×2) tiles × (3×4) tiles")
    
    # Both hierarchical (default)
    render_layout_svg(layout3, "assets/structure_spec3_both_hierarchical.svg")
    print("✓ Saved both hierarchical to assets/structure_spec3_both_hierarchical.svg")
    
    # Completely flat
    render_layout_svg(layout3, "assets/structure_spec3_flat.svg", coordinate_structure=(0, 0))
    print("✓ Saved completely flat to assets/structure_spec3_flat.svg")
    
    # First hierarchical, second flat
    render_layout_svg(layout3, "assets/structure_spec3_first_hier.svg", coordinate_structure=((0, 0), 0))
    print("✓ Saved first hierarchical to assets/structure_spec3_first_hier.svg")
    
    # First flat, second hierarchical
    render_layout_svg(layout3, "assets/structure_spec3_second_hier.svg", coordinate_structure=(0, (0, 0)))
    print("✓ Saved second hierarchical to assets/structure_spec3_second_hier.svg")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nSummary of coordinate_structure options:")
    print("  Simple values:")
    print("    None (default) → Hierarchical visualization (natural structure)")
    print("    0 or (0,)      → 1D visualization (single row)")
    print("    (0, 0)         → 2D visualization (flattened grid)")
    print("\n  Structure specifications (mirror layout structure):")
    print("    0 at position  → Flatten that mode")
    print("    Nested tuple   → Preserve hierarchical boundaries")
    print("\n  Examples for ((2,2), (3,4)):")
    print("    None           → Both modes hierarchical (default)")
    print("    (0, 0)         → Completely flat")
    print("    ((0,0), 0)     → First hierarchical, second flat")
    print("    (0, (0,0))     → First flat, second hierarchical")


if __name__ == "__main__":
    demonstrate_structure_specs()

