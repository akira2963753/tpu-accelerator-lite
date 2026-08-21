/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    TESTBED.sv
* Project:      tpu-accelerator-lite
* Module:       TESTBED (TSC standalone)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module TESTBED();

    initial begin
        $display("======================================");
        $display("  [INFO] BEHAVIORAL SIMULATION START  ");
        $display("======================================");
    end

    `ifdef DUMP
        initial begin
            $fsdbDumpfile("TESTBED.fsdb");
            $fsdbDumpvars(0, TESTBED, "+mda");
        end
    `endif

    //=============================================================
    // --------------------- Design & Pattern ---------------------
    //=============================================================
    logic                   clk, rst_n;
    logic                   cmd_valid, cmd_ready;
    logic [`CMD_DESC_W-1:0] cmd_desc;
    logic                   busy, done;
    logic                   w_valid, w_ready, a_valid, a_ready, r_valid, r_ready;
    // Datapath control
    logic wpu_wmem_w, wmem_en_c, wmem_en_w;
    logic [`WMEM_ADDR_W-1:0] wmem_addr;
    logic [`AMEM_ADDR_W-1:0] ub_addr_w, ub_addr_r;
    logic ub_en_w, ub_en_r;
    logic [`OBUF_ADDR_W-1:0] obuf_addr_w, obuf_addr_r;
    logic obuf_en_w, obuf_en_r, act_sel;
    logic weight_valid, skew_en, comp_clr, csa_cw_valid, csa_act_valid;
    logic acc_clr, cacc_en, acc_rd_en;
    logic [`ARRAY_S-1:0] acc_wr_en;
    logic [`ARRAY_S*`ROW_IDX_W-1:0] acc_wr_row;
    logic [`ROW_IDX_W-1:0] cacc_wr_row, acc_rd_row;
    logic [`QMULT_W-1:0]  quant_mult_q;
    logic [`QSHIFT_W-1:0] quant_shift_q;

    PATTERN u_pattern (
        .clk         (clk),
        .rst_n       (rst_n),
        .cmd_valid   (cmd_valid),
        .cmd_desc    (cmd_desc),
        .w_valid     (w_valid),
        .a_valid     (a_valid),
        .r_ready     (r_ready),
        .cmd_ready   (cmd_ready),
        .busy        (busy),
        .done        (done),
        .w_ready     (w_ready),
        .a_ready     (a_ready),
        .r_valid     (r_valid),
        .ub_addr_w   (ub_addr_w),
        .ub_addr_r   (ub_addr_r),
        .ub_en_w     (ub_en_w),
        .ub_en_r     (ub_en_r),
        .obuf_addr_w (obuf_addr_w),
        .obuf_addr_r (obuf_addr_r),
        .obuf_en_w   (obuf_en_w),
        .obuf_en_r   (obuf_en_r),
        .act_sel     (act_sel),
        .acc_clr     (acc_clr)
    );

    TSC u_dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (cmd_valid),
        .cmd_ready     (cmd_ready),
        .cmd_desc      (cmd_desc),
        .busy          (busy),
        .done          (done),
        .w_valid       (w_valid),
        .w_ready       (w_ready),
        .a_valid       (a_valid),
        .a_ready       (a_ready),
        .r_valid       (r_valid),
        .r_ready       (r_ready),
        .wpu_wmem_w    (wpu_wmem_w),
        .wmem_addr     (wmem_addr),
        .wmem_en_c     (wmem_en_c),
        .wmem_en_w     (wmem_en_w),
        .ub_addr_w     (ub_addr_w),
        .ub_addr_r     (ub_addr_r),
        .ub_en_w       (ub_en_w),
        .ub_en_r       (ub_en_r),
        .obuf_addr_w   (obuf_addr_w),
        .obuf_addr_r   (obuf_addr_r),
        .obuf_en_w     (obuf_en_w),
        .obuf_en_r     (obuf_en_r),
        .act_sel       (act_sel),
        .weight_valid  (weight_valid),
        .skew_en       (skew_en),
        .comp_clr      (comp_clr),
        .csa_cw_valid  (csa_cw_valid),
        .csa_act_valid (csa_act_valid),
        .acc_clr       (acc_clr),
        .acc_wr_en     (acc_wr_en),
        .acc_wr_row    (acc_wr_row),
        .cacc_en       (cacc_en),
        .cacc_wr_row   (cacc_wr_row),
        .acc_rd_en     (acc_rd_en),
        .acc_rd_row    (acc_rd_row),
        .quant_mult_q  (quant_mult_q),
        .quant_shift_q (quant_shift_q)
    );

endmodule
