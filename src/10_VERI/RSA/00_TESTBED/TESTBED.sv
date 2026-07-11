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
            $sdf_annotate("../02_SYN/Netlist/RSA_syn.sdf", u_dut, , ,"maximum");
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
    logic weight_valid;
    logic [`W_BW-1:0] weight;
    logic [`A_BW-1:0] activation;
    logic [`PSUM_BW-1:0] psum;

    PATTERN u_pattern (
        .clk(clk),
        .rst_n(rst_n),
        .weight_valid(weight_valid),
        .weight(weight),
        .activation(activation),
        .psum(psum)
    );

    RSA u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .weight_valid(weight_valid),
        .weight(weight),
        .activation(activation),
        .psum(psum)
    );

endmodule
