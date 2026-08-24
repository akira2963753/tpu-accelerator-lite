/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    TESTBED.sv
* Project:      tpu-accelerator-lite
* Module:       TESTBED (TPU top)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/
`include "define.vh"

module TESTBED;

    //=============================================================
    //                   Sim Mode & SDF Annotate
    //=============================================================
    `ifdef GATE
        initial begin
            $display("======================================");
            $display("  [INFO] GATE-LEVEL SIMULATION START  ");
            $display("======================================");
            $sdf_annotate("../02_SYN/Netlist/TPU_syn.sdf", u_dut, , ,"maximum");
        end
    `else
        initial begin
            $display("======================================");
            $display("  [INFO] BEHAVIORAL SIMULATION START  ");
            $display("======================================");
        end
    `endif

    //=============================================================
    //                          FSDB Dump
    //=============================================================
    // Dump waves when needed.
    `ifdef DUMP
        initial begin
            $fsdbDumpfile("TESTBED.fsdb");
            $fsdbDumpvars(0, TESTBED, "+mda");
        end
    `endif

    //=============================================================
    //                      Design & Pattern
    //=============================================================
    logic clk, rst_n;
    logic cmd_valid, cmd_ready;
    logic [`CMD_DESC_W-1:0] cmd_desc;
    logic w_valid, w_ready;
    logic [`W_RAW_BW-1:0] w_data;
    logic a_valid, a_ready;
    logic [`A_BW-1:0] a_data;
    logic r_valid, r_ready;
    logic [`R_BW-1:0] r_data;
    logic busy, done;

    PATTERN u_pattern (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_desc(cmd_desc),
        .w_valid(w_valid),
        .w_ready(w_ready),
        .w_data(w_data),
        .a_valid(a_valid),
        .a_ready(a_ready),
        .a_data(a_data),
        .r_valid(r_valid),
        .r_ready(r_ready),
        .r_data(r_data),
        .busy(busy),
        .done(done)
    );

    TPU u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_desc(cmd_desc),
        .w_valid(w_valid),
        .w_ready(w_ready),
        .w_data(w_data),
        .a_valid(a_valid),
        .a_ready(a_ready),
        .a_data(a_data),
        .r_valid(r_valid),
        .r_ready(r_ready),
        .r_data(r_data),
        .busy(busy),
        .done(done)
    );

endmodule
