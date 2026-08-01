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
            $sdf_annotate("../02_SYN/Netlist/WPU_syn.sdf", u_dut, , ,"maximum");
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
    logic clk, rst_n;
    logic [`W_RAW_BW-1:0] weight;
    logic [`WMEM_ADDR_W-1:0] wmem_addr_i;
    logic wmem_w;
    logic [`W_BW-1:0] rweight;
    logic [`CW_BW-1:0] cweight;
    logic [`ARRAY_S-1:0] cout_valid;
    logic [`ROW_IDX_W-1:0] crow;
    logic [`WMEM_ADDR_W-1:0] wmem_addr_o;

    PATTERN u_pattern (
        .clk(clk),
        .rst_n(rst_n),
        .weight(weight),
        .wmem_addr_i(wmem_addr_i),
        .wmem_w(wmem_w),
        .rweight(rweight),
        .cweight(cweight),
        .cout_valid(cout_valid),
        .crow(crow),
        .wmem_addr_o(wmem_addr_o)
    );

    WPU u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .weight(weight),
        .wmem_addr_i(wmem_addr_i),
        .wmem_w(wmem_w),
        .rweight(rweight),
        .cweight(cweight),
        .cout_valid(cout_valid),
        .crow(crow),
        .wmem_addr_o(wmem_addr_o)
    );

endmodule
