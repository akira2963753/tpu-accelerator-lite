/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    TESTBED.sv
* Project:      tpu-accelerator-lite
* Module:       TESTBED
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module TESTBED();
    
    //=============================================================
    // ---------------- Sim Mode & SDF Annotate -------------------
    //=============================================================
    `ifdef GATE
        initial begin
            $display("======================================");
            $display("  [INFO] GATE-LEVEL SIMULATION START  ");
            $display("======================================");
            $sdf_annotate("../02_SYN/Netlist/WMEM_Wrapper_syn.sdf", u_dut, , ,"maximum");
        end
    `else
        initial begin
            $display("======================================");
            $display("  [INFO] BEHAVIORAL SIMULATION START  ");
            $display("======================================");
        end
    `endif

    //=============================================================
    // ------------------------- FSDB Dump ------------------------
    //=============================================================
    initial begin
        $fsdbDumpfile("TESTBED.fsdb");
        $fsdbDumpvars(0, TESTBED, "+mda");
    end
    //=============================================================
    // --------------------- Design & Pattern ---------------------
    //=============================================================
    logic clk;
    logic [`WMEM_ADDR_W-1:0] addr_w;
    logic [`WMEM_DW-1:0] data_i;
    logic en_c;
    logic en_w;
    logic [`WMEM_DW-1:0] data_o;

    PATTERN u_pattern (
        .clk(clk),
        .addr_w(addr_w),
        .data_i(data_i),
        .en_c(en_c),
        .en_w(en_w),
        .data_o(data_o)
    );

    WMEM_Wrapper u_dut (
        .clk(clk),
        .addr_w(addr_w),
        .data_i(data_i),
        .en_c(en_c),
        .en_w(en_w),
        .data_o(data_o)
    );

endmodule
