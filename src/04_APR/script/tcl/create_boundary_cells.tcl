create_boundary_cells \
-left_boundary_cell    [get_lib_cells {N16ADFP_StdCell/BOUNDARY_LEFTBWP20P90LVT}] \
-right_boundary_cell   [get_lib_cells {N16ADFP_StdCell/BOUNDARY_RIGHTBWP20P90LVT}] \
-top_boundary_cells    [get_lib_cells {N16ADFP_StdCell/BOUNDARY_NROW1BWP20P90LVT \
                                       N16ADFP_StdCell/BOUNDARY_NROW2BWP20P90LVT \
                                       N16ADFP_StdCell/BOUNDARY_NROW3BWP20P90LVT \
                                       N16ADFP_StdCell/BOUNDARY_NROW4BWP20P90LVT}] \
-bottom_boundary_cells [get_lib_cells {N16ADFP_StdCell/BOUNDARY_PROW1BWP20P90LVT \
                                       N16ADFP_StdCell/BOUNDARY_PROW2BWP20P90LVT \
                                       N16ADFP_StdCell/BOUNDARY_PROW3BWP20P90LVT \
                                       N16ADFP_StdCell/BOUNDARY_PROW4BWP20P90LVT}] \
-top_left_outside_corner_cell     [get_lib_cells {N16ADFP_StdCell/BOUNDARY_NCORNERBWP20P90LVT}] \
-top_right_outside_corner_cell    [get_lib_cells {N16ADFP_StdCell/BOUNDARY_NCORNERBWP20P90LVT}] \
-bottom_left_outside_corner_cell  [get_lib_cells {N16ADFP_StdCell/BOUNDARY_PCORNERBWP20P90LVT}] \
-bottom_right_outside_corner_cell [get_lib_cells {N16ADFP_StdCell/BOUNDARY_PCORNERBWP20P90LVT}] \
-bottom_left_inside_corner_cells  [get_lib_cells {N16ADFP_StdCell/FILL3BWP20P90LVT}] \
-bottom_right_inside_corner_cells [get_lib_cells {N16ADFP_StdCell/FILL3BWP20P90LVT}] \
-top_left_inside_corner_cells     [get_lib_cells {N16ADFP_StdCell/FILL3BWP20P90LVT}] \
-top_right_inside_corner_cells    [get_lib_cells {N16ADFP_StdCell/FILL3BWP20P90LVT}] \
-bottom_tap_cell                  [get_lib_cells {N16ADFP_StdCell/BOUNDARY_PTAPBWP20P90LVT}] \
-top_tap_cell                     [get_lib_cells {N16ADFP_StdCell/BOUNDARY_NTAPBWP20P90LVT}] \
-tap_distance 50.7600 -mirror_left_outside_corner_cell
