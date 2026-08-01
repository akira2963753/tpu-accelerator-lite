#=================================================================
#---------- Synopsys Design Compiler Synthesis Scripts -----------
#=================================================================

#=================================================================
#--------------------- TOP Module Definition ---------------------
#=================================================================
set DESIGN  "TPU"

#=================================================================
#------------- Create the Working and Saving Folders -------------
#=================================================================
sh mkdir -p Netlist
sh mkdir -p Report
sh mkdir -p Work
define_design_lib $DESIGN -path Work

#=================================================================
#----------------- Analyze and Elaborate Design ------------------
#=================================================================
analyze -f sverilog -vcs "-f file.f"
elaborate $DESIGN
current_design $DESIGN

#=================================================================
#------------------- Set Operating Conditions --------------------
#=================================================================
set_operating_conditions -max_library N16ADFP_StdCellss0p72vm40c_ccs -max ss0p72vm40c \
    -min_library N16ADFP_StdCellff0p88vm125c_ccs -min ff0p88vm125c

source constraint.sdc
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
report_timing                        > Report/${DESIGN}_syn.timing
report_area -hierarchy               > Report/${DESIGN}_syn.area
report_power -hierarchy              > Report/${DESIGN}_syn.power
report_qor                           > Report/${DESIGN}_syn.qor
report_constraint -all_violators     > Report/${DESIGN}_syn.violators
report_reference                     > Report/${DESIGN}_syn.reference

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
