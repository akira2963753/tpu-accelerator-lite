/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    TESTBED.sv
* Project:      tpu-accelerator-lite
* Module:       TESTBED (TPU top)
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
    // ------------------------- FSDB Dump ------------------------
    //=============================================================
    // Gated : the full suite runs ~3.2M cycles (26M with --big) and dumping the
    // whole hierarchy with +mda (ACC 16x16xACC_W, Data_Setup 16x16 skew) writes
    // tens of GB. Build with +define+DUMP only when you actually need waves.
    `ifdef DUMP
        initial begin
            $fsdbDumpfile("TESTBED.fsdb");
            $fsdbDumpvars(0, TESTBED, "+mda");
        end
    `endif

    //=============================================================
    // --------------------- Design & Pattern ---------------------
    //=============================================================
    logic clk, rst_n;
    logic cmd_valid, cmd_ready;
    logic [`M_W-1:0] dim_m;
    logic [`K_W-1:0] dim_k;
    logic [`N_W-1:0] dim_n;
    logic [`QMULT_W-1:0] quant_mult;
    logic [`QSHIFT_W-1:0] quant_shift;
    logic fuse_in, fuse_out;
    logic w_valid, w_ready;
    logic [`W_RAW_BW-1:0] w_data;
    logic a_valid, a_ready;
    logic [`A_BW-1:0] a_data;
    logic r_valid, r_ready;
    logic [`A_BW-1:0] r_data;
    logic busy, done;

    PATTERN u_pattern (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .dim_m(dim_m),
        .dim_k(dim_k),
        .dim_n(dim_n),
        .quant_mult(quant_mult),
        .quant_shift(quant_shift),
        .fuse_in(fuse_in),
        .fuse_out(fuse_out),
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
        .dim_m(dim_m),
        .dim_k(dim_k),
        .dim_n(dim_n),
        .quant_mult(quant_mult),
        .quant_shift(quant_shift),
        .fuse_in(fuse_in),
        .fuse_out(fuse_out),
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
