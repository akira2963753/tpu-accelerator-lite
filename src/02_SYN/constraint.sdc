#=================================================================
#------------------------- Create Clock --------------------------
#=================================================================
set CYCLE   2

create_clock -name clk -period $CYCLE [get_ports clk]
set_dont_touch [all_clocks]
set_ideal_network [all_clocks]
# set_fix_hold [all_clocks]
set_cost_priority -min_delay


# Clock Constraints
set_clock_uncertainty -setup 0.5 -hold 0.005 [all_clocks]
set_clock_latency 0.5 [all_clocks]
set_clock_latency -source 0 [all_clocks]
set_clock_transition 0.2 [all_clocks]

#=================================================================
#---------------------- Timing Constraints -----------------------
#=================================================================

# Input Delay Constraints
set_input_delay [expr $CYCLE * 0.5] -clock clk [all_inputs]

# Output Delay Constraints
set_output_delay [expr $CYCLE * 0.5] -clock clk [all_outputs]

# Input Transition Constraints
set_input_transition 0.5 [all_inputs]