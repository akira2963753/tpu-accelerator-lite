#=================================================================
#---------- Synopsys Design Compiler Synthesis Scripts -----------
#=================================================================

#=================================================================
#-------------------- Set 16 nm Library Path ---------------------
#=================================================================
set search_path    "/usr/cad/ADFP/Executable_Package/Collaterals/IP/stdcell/N16ADFP_StdCell/CCS/ \
                    /usr/cad/ADFP/Executable_Package/Collaterals/IP/stdio/N16ADFP_StdIO/NLDM/ \
                    /usr/cad/ADFP/Executable_Package/Collaterals/IP/sram/N16ADFP_SRAM/NLDM/ \
                    $search_path .\
                    "

set target_library "N16ADFP_StdCellff0p88v125c_ccs.db \
                    N16ADFP_StdCellff0p88vm40c_ccs.db \
                    N16ADFP_StdCellss0p72v125c_ccs.db \
                    N16ADFP_StdCellss0p72vm40c_ccs.db \
                    N16ADFP_StdCelltt0p8v25c_ccs.db \
                    N16ADFP_StdIOff0p88v1p98v125c.db \
                    N16ADFP_StdIOff0p88v1p98vm40c.db \
                    N16ADFP_StdIOss0p72v1p62v125c.db \
                    N16ADFP_StdIOss0p72v1p62vm40c.db \
                    N16ADFP_StdIOtt0p8v1p8v25c.db \
                    N16ADFP_SRAM_ff0p88v0p88v125c_100a.db \
                    N16ADFP_SRAM_ff0p88v0p88vm40c_100a.db \
                    N16ADFP_SRAM_ss0p72v0p72v125c_100a.db \
                    N16ADFP_SRAM_ss0p72v0p72vm40c_100a.db \
                    N16ADFP_SRAM_tt0p8v0p8v25c_100a.db \
                    "

set link_library "* $target_library dw_foundation.sldb"
set symbol_library "generic.sdb"
set synthetic_library "dw_foundation.sldb"

#=================================================================
#---------- Global Setting and Environment Optimization ----------
#=================================================================
set hdlin_auto_save_templates true
set hdlin_check_no_latch true
set verilogout_no_tri true
set sh_enable_line_editing true
history keep 1000
alias h history
set sh_continue_on_error false
set compile_preserve_subdesign_interfaces true

#=================================================================
#--------------------- TOP Module Definition ---------------------
#=================================================================
set DESIGN  "TPU"
set CYCLE 2

#=================================================================
#------------- Create the Working and Saving Folders -------------
#=================================================================
sh mkdir -p Netlist
sh mkdir -p Report
sh mkdir -p Work
define_design_lib WORK -path Work

#=================================================================
#----------------- Analyze and Elaborate Design ------------------
#=================================================================
analyze -f sverilog -vcs "-f file.f"
elaborate $DESIGN
current_design $DESIGN

#=================================================================
#------------------- Set Operating Conditions --------------------
#=================================================================
set_operating_conditions \
    -max_library N16ADFP_StdCellss0p72vm40c_ccs \
    -max ss0p72vm40c \
    -min_library N16ADFP_StdCellff0p88v125c_ccs \
    -min ff0p88vm125c

#=================================================================
#------------------------- Create Clock --------------------------
#=================================================================
create_clock -name clk -period $CYCLE [get_ports clk]
set_dont_touch [all_clocks]
set_ideal_network [all_clocks]
set_fix_hold [all_clocks]
set_clock_uncertainty -setup 0.5 [all_clocks] 
set_clock_uncertainty -hold 0.005 [all_clocks]
set_clock_latency 0.5 [all_clocks]
set_clock_latency -source 0 [all_clocks]
set_clock_transition 0.2 [all_clocks]

#=================================================================
#---------------------- Timing Constraints -----------------------
#=================================================================
set_input_delay [expr $CYCLE * 0.5] -clock clk [all_inputs]
set_output_delay [expr $CYCLE * 0.5] -clock clk [all_outputs]

#=================================================================
#-------------------- Design Rule Constraints --------------------
#=================================================================

set_driving_cell -library "N16ADFP_StdCellss0p72vm40c_ccs" -lib_cell BUFFD4BWP16P90LVT -pin {Z} [get_ports clk]
set_driving_cell -library "N16ADFP_StdCellss0p72vm40c_ccs" -lib_cell DFQD1BWP16P90LVT -pin {Q} [remove_from_collection [all_inputs] [get_ports clk]]
set_load [load_of "N16ADFP_StdCellss0p72vm40c_ccs/DFQD1BWP16P90LVT/D"] [all_outputs]

set_max_area 0
set_max_capacitance 0.1 [all_inputs]
set_max_fanout 10 [all_inputs]
set_max_transition 0.2 [all_inputs]

write -format ddc -hierarchy -output "./Netlist/Pre_Compile_${DESIGN}.ddc"
#=================================================================
#-------------------- Compile & Optimization ---------------------
#=================================================================
uniquify
set_fix_multiple_port_nets -all -buffer_constants [get_designs *]
current_design $DESIGN
compile_ultra

#=================================================================
#------------------------ Report & Output ------------------------
#=================================================================
current_design $DESIGN
report_timing > Report/${DESIGN}_syn.timing
report_area -hierarchy > Report/${DESIGN}_syn.area
report_power -hierarchy > Report/${DESIGN}_syn.power
report_qor > Report/${DESIGN}_syn.qor
report_constraint -all_violators > Report/${DESIGN}_syn.violators
report_reference > Report/${DESIGN}_syn.reference

set bus_inference_style {%s[%d]}
set bus_naming_style {%s[%d]}
set hdlout_internal_busses true

change_names -hierarchy -rule verilog
define_name_rules name_rule -allowed {a-z A-Z 0-9 _} -max_length 255 -type cell
define_name_rules name_rule -allowed {a-z A-Z 0-9 _[]} -max_length 255 -type net
define_name_rules name_rule -map {{"\\*cell\\*" "cell"}}
define_name_rules name_rule -case_insensitive
change_names -hierarchy -rules name_rule

remove_unconnected_ports -blast_buses [get_cells -hierarchical *]
set verilogout_higher_designs_first true
write -format ddc -hierarchy -output "./Netlist/${DESIGN}.ddc"
write -format verilog -hierarchy -output "./Netlist/${DESIGN}_syn.v"
write_sdf ./Netlist/${DESIGN}_syn.sdf
write_sdc ./Netlist/${DESIGN}_syn.sdc

report_timing
report_area
