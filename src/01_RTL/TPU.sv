/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    TPU.sv
* Project:      tpu-accelerator-lite
* Module:       TPU (Tensor Processing Unit)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module TPU (
    input clk,
    input rst_n,
    // job descriptor + start handshake
    input cmd_valid,
    output cmd_ready,
    input [`M_W-1:0] dim_m,
    input [`K_W-1:0] dim_k,
    input [`N_W-1:0] dim_n,
    // per-tensor requant constants (latched with the descriptor)
    input [`QMULT_W-1:0] quant_mult,
    input [`QSHIFT_W-1:0] quant_shift,
    // weight / activation load (valid/ready)
    input w_valid,
    output w_ready,
    input [`W_RAW_BW-1:0] w_data, // raw weight row
    input a_valid,
    output a_ready,
    input [`A_BW-1:0] a_data, // activation row
    // result output (valid/ready)
    output r_valid,
    input r_ready,
    output [`A_BW-1:0] r_data,
    // status
    output busy,
    output done
);

    // ---- control (TSC -> datapath) ----
    logic wpu_wmem_w;
    logic [`WMEM_ADDR_W-1:0] wmem_addr;
    logic wmem_en_c, wmem_en_w;
    logic [`AMEM_ADDR_W-1:0] ub_addr_w, ub_addr_r;
    logic ub_en_w, ub_en_r;
    logic weight_valid, skew_en, comp_clr, csa_cw_valid, csa_act_valid;
    logic acc_clr;
    logic [`ARRAY_S-1:0] acc_wr_en;
    logic [`ARRAY_S*`ROW_IDX_W-1:0] acc_wr_row;
    logic cacc_en;
    logic [`ROW_IDX_W-1:0] cacc_wr_row;
    logic acc_rd_en;
    logic [`ROW_IDX_W-1:0] acc_rd_row;

    // ---- datapath nets ----
    logic [`W_BW-1:0] wpu_rweight;
    logic [`CW_BW-1:0] wpu_cweight;
    logic [`ARRAY_S-1:0] wpu_cout_valid;
    logic [`ROW_IDX_W-1:0] wpu_crow;
    logic [`WMEM_ADDR_W-1:0] wpu_wmem_addr_o;
    logic [`WMEM_DW-1:0] wmem_data_o;
    logic [`CW_BW-1:0] cmem_cweight_o;
    logic [`AMEM_DW-1:0] ub_data_o;
    logic [`A_BW-1:0] ds_activation_o;
    logic [`A_BW-1:0] ds_activation_cout_o;
    logic [`ARRAY_S-1:0] ds_cout_valid_o;
    logic [`PSUM_BW-1:0] rsa_psum;
    logic [`ARRAY_S*`CPSUM_W-1:0] csa_cpsum_o;
    logic [`ARRAY_S*`ACC_W-1:0] acc_o;

    // WMEM address : write uses WPU's registered address, preload read uses TSC's
    logic [`WMEM_ADDR_W-1:0] wmem_addr_w;
    assign wmem_addr_w = (wmem_en_w) ? wpu_wmem_addr_o : wmem_addr;

    // UB write data : external activation during load, result write-back during OUT
    // (r_valid is high only in OUT, so it doubles as the write-back select)
    logic [`A_BW-1:0] ub_wdata;
    assign ub_wdata = (r_valid)? r_data : a_data;

    // requant constants held by TSC (latched with the descriptor there), driven to OP
    logic [`QMULT_W-1:0] qmult_q;
    logic [`QSHIFT_W-1:0] qshift_q;

    // ================= control =================
    TSC u_TSC(
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .dim_m(dim_m),
        .dim_k(dim_k),
        .dim_n(dim_n),
        .quant_mult(quant_mult),
        .quant_shift(quant_shift),
        .busy(busy),
        .done(done),
        .w_valid(w_valid),
        .w_ready(w_ready),
        .a_valid(a_valid),
        .a_ready(a_ready),
        .r_valid(r_valid),
        .r_ready(r_ready),
        .wpu_wmem_w(wpu_wmem_w),
        .wmem_addr(wmem_addr),
        .wmem_en_c(wmem_en_c),
        .wmem_en_w(wmem_en_w),
        .ub_addr_w(ub_addr_w),
        .ub_addr_r(ub_addr_r),
        .ub_en_w(ub_en_w),
        .ub_en_r(ub_en_r),
        .weight_valid(weight_valid),
        .skew_en(skew_en),
        .comp_clr(comp_clr),
        .csa_cw_valid(csa_cw_valid),
        .csa_act_valid(csa_act_valid),
        .acc_clr(acc_clr),
        .acc_wr_en(acc_wr_en),
        .acc_wr_row(acc_wr_row),
        .cacc_en(cacc_en),
        .cacc_wr_row(cacc_wr_row),
        .acc_rd_en(acc_rd_en),
        .acc_rd_row(acc_rd_row),
        .quant_mult_q(qmult_q),
        .quant_shift_q(qshift_q));

    // ================= weight path =================
    WPU u_WPU(
        .clk(clk),
        .rst_n(rst_n),
        .weight(w_data),
        .wmem_addr_i(wmem_addr),
        .wmem_w(wpu_wmem_w),
        .rweight(wpu_rweight),
        .cweight(wpu_cweight),
        .cout_valid(wpu_cout_valid),
        .crow(wpu_crow),
        .wmem_addr_o(wpu_wmem_addr_o));

    WMEM_Wrapper u_WMEM_Wrapper(
        .clk(clk),
        .addr_w(wmem_addr_w),
        .data_i(wpu_rweight),
        .en_c(wmem_en_c),
        .en_w(wmem_en_w),
        .data_o(wmem_data_o));

    CMEM u_CMEM(
        .clk(clk),
        .rst_n(rst_n),
        .cout_valid(wpu_cout_valid),
        .cweight(wpu_cweight),
        .cweight_o(cmem_cweight_o));

    // ================= activation path =================
    UB_Wrapper u_UB_Wrapper(
        .clk(clk),
        .data_i(ub_wdata),
        .addr_w(ub_addr_w),
        .addr_r(ub_addr_r),
        .en_w(ub_en_w),
        .en_r(ub_en_r),
        .data_o(ub_data_o));

    Data_Setup u_Data_Setup(
        .clk(clk),
        .rst_n(rst_n),
        .activation_i(ub_data_o),
        .skew_en(skew_en),
        .comp_clr(comp_clr),
        .cout_valid(wpu_cout_valid),
        .crow(wpu_crow),
        .activation_o(ds_activation_o),
        .activation_cout_o(ds_activation_cout_o),
        .cout_valid_o(ds_cout_valid_o));

    // ================= compute =================
    RSA u_RSA(
        .clk(clk),
        .rst_n(rst_n),
        .weight_valid(weight_valid),
        .weight(wmem_data_o),
        .activation(ds_activation_o),
        .psum(rsa_psum));

    CSA u_CSA(
        .clk(clk),
        .rst_n(rst_n),
        .cweight(cmem_cweight_o),
        .cweight_valid(csa_cw_valid),
        .activation(ds_activation_cout_o),
        .activation_valid(csa_act_valid),
        .cpsum_o(csa_cpsum_o));

    // ================= accumulate + output =================
    ACC u_ACC(
        .clk(clk),
        .rst_n(rst_n),
        .acc_clr(acc_clr),
        .psum_i(rsa_psum),
        .cpsum_i(csa_cpsum_o),
        .cout_valid(ds_cout_valid_o),
        .acc_wr_en(acc_wr_en),
        .acc_wr_row(acc_wr_row),
        .cacc_en(cacc_en),
        .cacc_wr_row(cacc_wr_row),
        .acc_rd_en(acc_rd_en),
        .acc_rd_row(acc_rd_row),
        .acc_o(acc_o));

    OP u_OP(
        .quant_mult(qmult_q),
        .quant_shift(qshift_q),
        .acc_i(acc_o),
        .data_o(r_data));

endmodule
