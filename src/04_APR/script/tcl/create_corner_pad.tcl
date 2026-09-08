create_cell {cornerBL cornerTL cornerTR cornerBR} N16ADFP_StdIO/PCORNER
create_io_corner_cell -cell cornerBL {ioring.bottom ioring.left}
create_io_corner_cell -cell cornerTL {ioring.left ioring.top}
create_io_corner_cell -cell cornerTR {ioring.top ioring.right}
create_io_corner_cell -cell cornerBR {ioring.right ioring.bottom}
set_attribute -objects [get_cells cornerBR] -name orientation -value MY
set_attribute -objects [get_cells cornerTL] -name orientation -value MX
