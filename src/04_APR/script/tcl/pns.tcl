connect_pg_net
#connect_pg_net -net VDDPST [get_pins */VDDPST]
################################
# create power ring
################################
create_pg_ring_pattern ring_pattern_3_4 \
                      -nets {VDD VSS} \
                      -horizontal_layer M4 -vertical_layer M3 \
                      -horizontal_width {2} -vertical_width {2} \
                      -horizontal_spacing {2} -vertical_spacing {2} \
                      -corner_bridge true

set_pg_strategy Strategy_ring_3_4 -core -pattern {{name : ring_pattern_3_4}{nets : {VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD}}{offset : {15 15}}}

create_pg_ring_pattern ring_pattern_5_6 \
                      -nets {VDD VSS} \
                      -horizontal_layer M6 -vertical_layer M5 \
                      -horizontal_width {2} -vertical_width {2} \
                      -horizontal_spacing {2} -vertical_spacing {2} \
                      -corner_bridge true

set_pg_strategy Strategy_ring_5_6 -core -pattern {{name : ring_pattern_5_6}{nets : {VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD}}{offset : {15 15}}}

create_pg_ring_pattern ring_pattern_7_8 \
                      -nets {VDD VSS} \
                      -horizontal_layer M8 -vertical_layer M7 \
                      -horizontal_width {2} -vertical_width {2} \
                      -horizontal_spacing {2} -vertical_spacing {2} \
                      -corner_bridge true

set_pg_strategy Strategy_ring_7_8 -core -pattern {{name : ring_pattern_7_8}{nets : {VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD VSS VDD}}{offset : {15 15}}}

compile_pg -strategies {Strategy_ring_3_4 Strategy_ring_5_6 Strategy_ring_7_8}

################################
# connect power pad to power ring
################################
#connect_supply_net VDD -port [get_pins -hierarchical */VDD]
#connect_supply_net VSS -port [get_pins -hierarchical */VSS]
create_pg_macro_conn_pattern iopg_pattern_v -pin_conn_type scattered_pin -nets {VSS VDD} -layers {M3 M2} -pin_layers {M2}

create_pg_macro_conn_pattern iopg_pattern_h -pin_conn_type scattered_pin -nets {VSS VDD} -layers {M2 M1} -pin_layers {M2}

set_app_options -name plan.pgroute.treat_pad_as_macro -value true

set iopgs [get_cells core*]

set_pg_strategy Strategy_iopg_v -macros {core_power1 core_power2 core_power3 core_power4} -pattern {{name : iopg_pattern_v}{nets : {VDD VSS}}} -extension {{{stop : 168.888}}}

set_pg_strategy Strategy_iopg_h -macros {core_power1 core_power2 core_power3 core_power4} -pattern {{name : iopg_pattern_h}{nets : {VDD VSS}}} -extension {{{stop : 168.888}}}

compile_pg -strategies {Strategy_iopg_v Strategy_iopg_h}

create_pg_macro_conn_pattern iopg_pattern_v2 -pin_conn_type scattered_pin -nets {VSS VDD} -layers {M2 M3} -pin_layers {M3}

create_pg_macro_conn_pattern iopg_pattern_h2 -pin_conn_type scattered_pin -nets {VSS VDD} -layers {M4 M5} -pin_layers {M4}

set_pg_strategy Strategy_iopg_v2 -macros {core_power1 core_power2 core_power3 core_power4} -pattern {{name : iopg_pattern_v2}{nets : {VDD VSS}}} -extension {{{stop : 168.888}}}

set_pg_strategy Strategy_iopg_h2 -macros {core_power1 core_power2 core_power3 core_power4} -pattern {{name : iopg_pattern_h2}{nets : {VDD VSS}}} -extension {{{stop : 168.888}}}

compile_pg -strategies {Strategy_iopg_v2 Strategy_iopg_h2}

################################
# Create Macro Block ring
################################


######################################
# create power straps
######################################
create_pg_mesh_pattern mesh_pattern_5_8 -layers {{{vertical_layer : M5} {width : 1} {spacing : 1} {pitch : 10} {trim : true}} {{horizontal_layer : M6} {width : 1} {spacing : 1} {pitch : 10} {trim : true}} {{vertical_layer : M7} {width : 1} {spacing : 1} {pitch : 10} {trim : true}} {{horizontal_layer : M8} {width : 1} {spacing : 1} {pitch : 10} {trim : true}}}

set_pg_strategy Strategy_5_8 -core -pattern {{name : mesh_pattern_5_8}{nets : {VSS VDD}}} -extension {{{nets : {VSS VDD}}{stop : outermost_ring}}}

set_pg_strategy_via_rule via_rule_5_8 -via_rule {{{{macro_pins : all}{via_master : default}}}{{existing : ring}{via_master : default}}}

compile_pg -strategies Strategy_5_8 -via_rule via_rule_5_8

#create_pg_strap -layer M5 -direction vertical -width 1.000 -net VSS -start 237.364 -stop 237.364
#create_pg_strap -layer M7 -direction vertical -width 1.000 -net VSS -start 237.364 -stop 237.364
source ../scripts/create_strap.tcl

################################
# M5 straps over macro
################################


################################
# Create power rails
################################
create_pg_std_cell_conn_pattern rail_pattern -layers {M1 M2}

set_pg_strategy power_rails -core -pattern {{name : rail_pattern}{nets : {VSS VDD}}} -blockage {{{macros_with_keepout : {u_DCT/tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i0 u_DCT/tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i1 u_DCT/tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i2 u_DCT/tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i3 u_DCT_L2_tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i0 u_DCT_L2_tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i1 u_DCT_L2_tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i2 u_DCT_L2_tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i3 u_IDCT/tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i0 u_IDCT/tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i1 u_IDCT/tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i2 u_IDCT/tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i3 u_IDCT_L2_tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i0 u_IDCT_L2_tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i1 u_IDCT_L2_tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i2 u_IDCT_L2_tposemem_Bisted_RF_2P_ADV64x16_RF_2P_ADV64x16_u0_i_rf_2p_SRAM_i3}}} -extension {{{nets : {VSS VDD}}{stop : outermost_ring}}}

set_pg_strategy_via_rule via_rule_1_5 -via_rule {{{{existing : ring}{nets : {VDD VSS}}{layers : {M1 M2 M3 M4 M5 M6 M7 M8}}}{via_master : default}}}{{{existing : strap}{nets : {VDD VSS}}{layers : {M1 M2 M3 M4 M5}}}{via_master : default}}}

compile_pg -strategies power_rails -via_rule via_rule_1_5

check_pg_drc
check_pg_connectivity
check_pg_missing_vias

analyze_power_plan -nets {VDD VSS} -analyze_power -voltage 0.8
