#=================================================================
#---------- Synopsys Design Compiler Synthesis Scripts -----------
#=================================================================

#=================================================================
#-------------------- Set 90 nm Library Path ---------------------
#=================================================================
set search_path "/usr/cad/CBDK_TSMC90G_Arm_v1.1/Lib90 $search_path"
set target_library "slow.db fast.db typical.db tpzn90gv3bc.db tpzn90gv3wc.db"
set link_library "* $target_library dw_foundation.sldb"
set symbol_library "tsmc090.sdb"
set synthetic_library "dw_foundation.sldb"

#=================================================================
#--------------------- TOP Module Definition ---------------------
#=================================================================
set DESIGN  "WPU"
set CYCLE   15

#=================================================================
#------------- Create the Working and Saving Folders -------------
#=================================================================
sh mkdir -p Netlist
sh mkdir -p Report
sh mkdir -p Work
define_design_lib $DESIGN -path Work

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
#------------------- Set Operating Conditions --------------------
#=================================================================
set_operating_conditions -min fast -max slow

#=================================================================
#----------------- Analyze and Elaborate Design ------------------
#=================================================================
analyze -f sverilog -vcs "-f file.f"
elaborate $DESIGN
current_design $DESIGN

#=================================================================
#------------------------- Create Clock --------------------------
#=================================================================
create_clock -name clk -period $CYCLE [get_ports clk]
set_dont_touch [all_clocks]
set_ideal_network [all_clocks]
set_fix_hold [all_clocks]

# Clock Constraints
set_clock_uncertainty -hold 0.005 [all_clocks]
set_clock_uncertainty -setup 0.1 [all_clocks]
set_clock_latency 0.5 [all_clocks]
set_clock_latency -source 0 [all_clocks]
set_clock_transition 0.1 [all_clocks]

#=================================================================
#---------------------- Timing Constraints -----------------------
#=================================================================

# Input Delay Constraints
set_input_delay [expr $CYCLE * 0.5] -clock clk [all_inputs]

# Output Delay Constraints
set_output_delay [expr $CYCLE * 0.5] -clock clk [all_outputs]

# Input Transition Constraints
set_input_transition 0.2 [all_inputs]

# Max delay from input to output
set_max_delay 0 -from [all_inputs] -to [all_outputs]

#=================================================================
#-------------------- Design Rule Constraints --------------------
#=================================================================

#set_driving_cell -library tpzn90gv3wc -lib_cell PDIDGZ_33 -pin {C} [all_inputs]
#set_load [load_of "tpzn90gv3wc/PDO16CDG_33/I"] [all_outputs]

set_drive 1 [all_inputs]
set_load 0.05 [all_outputs]

set_max_capacitance 0.1 [all_inputs]
set_max_fanout 10 [all_inputs]
set_max_transition 0.2 [all_inputs]

#=================================================================
#-------------------------- False Path ---------------------------
#=================================================================

# set_false_path -from [get_clocks clk1] -to [get_clocks clk2]
# set_false_path -from [get_clocks clk2] -to [get_clocks clk1]

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
report_area > Report/${DESIGN}_syn.area

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
write -format ddc      -hierarchy -output "./Netlist/${DESIGN}.ddc"
write -format verilog  -hierarchy -output "./Netlist/${DESIGN}_syn.v"
write_sdf ./Netlist/${DESIGN}_syn.sdf
write_sdc ./Netlist/${DESIGN}_syn.sdc

report_timing
report_area
