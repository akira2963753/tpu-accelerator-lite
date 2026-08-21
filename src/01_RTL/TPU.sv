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

    input cmd_valid,
    output cmd_ready,
    input [`CMD_DESC_W-1:0] cmd_desc,

    input w_valid,
    output w_ready,
    input [`W_RAW_BW-1:0] w_data,
    input a_valid,
    output a_ready,
    input [`A_BW-1:0] a_data,

    output r_valid,
    input r_ready,
    output [`R_BW-1:0] r_data,

    output busy,
    output done
);

    logic wpu_wmem_w;
    logic [`WMEM_ADDR_W-1:0] wmem_addr;
    logic wmem_en, wmem_we;
    logic [`UB_ADDR_W-1:0] ub_addr;
    logic ub_en, ub_we;
    logic r_from_ub;
    logic weight_valid, skew_en, activation_valid;
    logic acc_clr;
    logic [`ARRAY_S-1:0] acc_wr_en;
    logic [`ARRAY_S*`ROW_IDX_W-1:0] acc_wr_row;
    logic acc_rd_en;
    logic [`ROW_IDX_W-1:0] acc_rd_row;

    logic [`W_BW-1:0] wpu_rweight;
    logic [`WMEM_ADDR_W-1:0] wpu_wmem_addr;
    logic [`WMEM_ADDR_W-1:0] wmem_mem_addr;
    logic [`WMEM_DW-1:0] wmem_data_o;
    logic [`UB_DW-1:0] ub_data_i;
    logic [`UB_DW-1:0] ub_data_o;
    logic [`A_BW-1:0] activation_row;
    logic [`A_BW-1:0] ds_activation_o;
    logic [`PSUM_BW-1:0] rsa_psum;
    logic [`ARRAY_S*`ACC_W-1:0] acc_o;
    logic [`A_BW-1:0] op_data;
    logic [`R_BW-1:0] op_r_data;
    logic [`R_BW-1:0] ub_r_data;
    logic [`QMULT_W-1:0] qmult_q;
    logic [`QSHIFT_W-1:0] qshift_q;

    assign wmem_mem_addr = wmem_we ? wpu_wmem_addr : wmem_addr;
    assign ub_data_i = (r_valid && !r_from_ub) ? op_data : a_data;
    assign activation_row = activation_valid ? ub_data_o : 0;
    assign r_data = r_from_ub ? ub_r_data : op_r_data;

    genvar i;
    generate
        for(i = 0; i < `ARRAY_S; i++) begin : OUTPUT_FORMAT
            assign op_r_data[`R_W*i +: `R_W] = {
                op_data[`A_W*i +: `A_W],
                {(`R_W-`A_W){1'b0}}
            };
            assign ub_r_data[`R_W*i +: `R_W] = {
                ub_data_o[`A_W*i +: `A_W],
                {(`R_W-`A_W){1'b0}}
            };
        end
    endgenerate

    TSC u_TSC(
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_desc(cmd_desc),
        .busy(busy),
        .done(done),
        .w_valid(w_valid),
        .w_ready(w_ready),
        .a_valid(a_valid),
        .a_ready(a_ready),
        .r_valid(r_valid),
        .r_from_ub(r_from_ub),
        .r_ready(r_ready),
        .wpu_wmem_w(wpu_wmem_w),
        .wmem_addr(wmem_addr),
        .wmem_en(wmem_en),
        .wmem_we(wmem_we),
        .ub_addr(ub_addr),
        .ub_en(ub_en),
        .ub_we(ub_we),
        .weight_valid(weight_valid),
        .skew_en(skew_en),
        .activation_valid(activation_valid),
        .acc_clr(acc_clr),
        .acc_wr_en(acc_wr_en),
        .acc_wr_row(acc_wr_row),
        .acc_rd_en(acc_rd_en),
        .acc_rd_row(acc_rd_row),
        .quant_mult_q(qmult_q),
        .quant_shift_q(qshift_q));

    WPU u_WPU(
        .clk(clk),
        .rst_n(rst_n),
        .weight(w_data),
        .wmem_addr_i(wmem_addr),
        .wmem_w(wpu_wmem_w),
        .rweight(wpu_rweight),
        .wmem_addr_o(wpu_wmem_addr));

    WMEM_Wrapper u_WMEM_Wrapper(
        .clk(clk),
        .addr(wmem_mem_addr),
        .data_i(wpu_rweight),
        .en(wmem_en),
        .we(wmem_we),
        .data_o(wmem_data_o));

    UB_Wrapper u_UB_Wrapper(
        .clk(clk),
        .addr(ub_addr),
        .data_i(ub_data_i),
        .en(ub_en),
        .we(ub_we),
        .data_o(ub_data_o));

    Data_Setup u_Data_Setup(
        .clk(clk),
        .rst_n(rst_n),
        .activation_i(activation_row),
        .skew_en(skew_en),
        .activation_o(ds_activation_o));

    RSA u_RSA(
        .clk(clk),
        .rst_n(rst_n),
        .weight_valid(weight_valid),
        .weight(wmem_data_o),
        .activation(ds_activation_o),
        .psum(rsa_psum));

    ACC u_ACC(
        .clk(clk),
        .rst_n(rst_n),
        .acc_clr(acc_clr),
        .psum_i(rsa_psum),
        .acc_wr_en(acc_wr_en),
        .acc_wr_row(acc_wr_row),
        .acc_rd_en(acc_rd_en),
        .acc_rd_row(acc_rd_row),
        .acc_o(acc_o));

    OP u_OP(
        .quant_mult(qmult_q),
        .quant_shift(qshift_q),
        .acc_i(acc_o),
        .data_o(op_data));

endmodule
