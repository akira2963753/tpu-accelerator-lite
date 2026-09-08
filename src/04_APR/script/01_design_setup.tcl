create_lib CHIP -ref_libs {NDM/ADFP_stdIO_physical_only.ndm NDM/N16ADFP_StdCell.ndm NDM/N16ADFP_StdIO.ndm NDM/sram.ndm} \
    -technology /usr/cad/ADFP/Executable_Package/Collaterals/Tech/APR/N16ADFP_APR_ICC2/N16ADFP_APR_ICC2_11M.10a.tf

report_ref_lib
read_verilog ./CHIP_syn.v -top CHIP
link_block

read_parasitic_tech -tlup /usr/cad/ADFP/Executable_Package/Collaterals/Tech/RC/N16ADFP_STARRC/N16ADFP_STARRC_worst.nxtgrd -name max
read_parasitic_tech -tlup /usr/cad/ADFP/Executable_Package/Collaterals/Tech/RC/N16ADFP_STARRC/N16ADFP_STARRC_best.nxtgrd -name min
report_lib -parasitic_tech [current_lib]
set_parasitic_parameters -early_spec min
set_parasitic_parameters -late_spec max

set_attribute [get_site_defs unit] is_default true
set_attribute [get_site_defs unit] symmetry Y
set_attribute [get_layers {M1}] track_offset 0.045
set_attribute [get_layers {M1 M3 M5 M7 M9 M11}] routing_direction vertical
set_attribute [get_layers {M2 M4 M6 M8 M10 AP}] routing_direction horizontal
report_ignored_layers

source ./CHIP_syn.sdc
set_timing_derate -late 1.02 -cell_delay [get_cells -hier *]

save_block -as CHIP:design_setup.design
save_lib
