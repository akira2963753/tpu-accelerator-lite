/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    TESTBED.sv
* Project:      tpu-accelerator-lite
* Module:       TESTBED (CHIP_TOP)
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
            $sdf_annotate("../02_SYN/Netlist/CHIP_TOP_syn.sdf", u_dut, , ,"maximum");
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
    logic [`HOST_DATA_W-1:0] host_data_i;
    logic [`HOST_TYPE_W-1:0] host_type;
    logic host_valid, host_ready;
    logic [`HOST_DATA_W-1:0] host_data_o;
    logic host_out_valid, host_out_ready;
    logic busy, done;

    PATTERN u_pattern (
        .clk(clk),
        .rst_n(rst_n),
        .host_data_i(host_data_i),
        .host_type(host_type),
        .host_valid(host_valid),
        .host_ready(host_ready),
        .host_data_o(host_data_o),
        .host_out_valid(host_out_valid),
        .host_out_ready(host_out_ready),
        .busy(busy),
        .done(done)
    );

    CHIP_TOP u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .host_data_i(host_data_i),
        .host_type(host_type),
        .host_valid(host_valid),
        .host_ready(host_ready),
        .host_data_o(host_data_o),
        .host_out_valid(host_out_valid),
        .host_out_ready(host_out_ready),
        .busy(busy),
        .done(done)
    );

endmodule
