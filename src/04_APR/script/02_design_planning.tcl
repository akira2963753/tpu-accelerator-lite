initialize_floorplan -core_utilization 0.6 -honor_pad_limit -core_offset {233.864}
create_io_ring -name ioring -corner_height 78.864

source ./script/tcl/create_corner_pad.tcl
source ./script/tcl/create_power_pad.tcl
set_signal_io_constraints -file ./CHIP.io

initialize_floorplan -core_utilization 0.6 -honor_pad_limit -core_offset {233.864}
create_io_ring -name ioring -corner_height 78.864
source ./script/tcl/create_corner_pad.tcl
set_signal_io_constraints -file ./CHIP.io

create_bump_array -lib_cell PAD80APB_LF_BU -delta {120 120} -origin {79.104 79.104}
place_io
create_io_filler_cells -reference_cells {PFILLER10080 PFILLER01008 PFILLER00048 PFILLER00001} -overlap_cells PFILLER00001

set io_insts [get_cells -hier -filter "is_io==true"]
set_fixed_objects $io_insts

save_block
save_block -as CHIP:die_init.design
save_lib

set all_macros [get_cells -hierarchical -filter "is_hard_macro && !is_physical_only"]
create_keepout_margin -type hard -outer {3 3 3 3} $all_macros
create_keepout_margin -type hard_macro -outer {12 12 12 12} $all_macros
create_keepout_margin -type routing_blockage -outer {1 1 1 1} -layers {M4 M5} $all_macros
source ./script/tcl/macro_app_options.tcl

create_placement -floorplan
check_finfet_grid
set_fixed_objects $all_macros

source ./script/tcl/create_boundary_cells.tcl
source ./script/tcl/create_tap_cells.tcl
create_placement -incremental

save_block
save_block -as CHIP:before_pns.design
save_lib

#source -echo ./script/tcl/pns.tcl
#create_placement -incremental

#save_block
#save_block -as CHIP:design_planning.design
#save_lib

#write_floorplan -net_types {power ground} \
#  -include_physical_status {fixed locked} \
#  -force -output CHIP_icc2.fp