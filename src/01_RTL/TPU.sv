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
    logic [`ROW_IDX_W-1:0] activation_row_idx;
    logic amask_init, amask_default_valid, amask_wr_en;
    logic [`AMASK_BEAT_W-1:0] amask_wr_beat;
    logic bias_wr_en;
    logic [`BIAS_BEAT_W-1:0] bias_wr_beat;
    logic acc_clr;
    logic acc_bias_en;
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
    logic [`AMASK_BW-1:0] active_amask;
    logic [`ARRAY_S-1:0] activation_mask_row;
    logic [`ARRAY_S-1:0] ds_activation_mask;
    logic [`PSUM_BW-1:0] rsa_psum;
    logic [`ARRAY_S*`ACC_W-1:0] acc_o;
    logic [`A_BW-1:0] op_data;
    logic [`R_BW-1:0] op_r_data;
    logic [`R_BW-1:0] ub_r_data;

    // TFLite's Quantize Multiplier:
    // Yi = (Wi * Ai) * (Sw * Sa) / Sy
    // M = Sw * Sa / Sy encode as qmult / 2^qshift

    logic [`QMULT_W-1:0] qmult_q;
    logic [`QSHIFT_W-1:0] qshift_q;
    logic [`K_VALID_W-1:0] k_valid_q;
    logic [`ACT_MODE_W-1:0] act_mode_q;

    assign wmem_mem_addr = (wmem_we)? wpu_wmem_addr : wmem_addr;
    assign ub_data_i = (r_valid && !r_from_ub)? op_data : a_data;
    assign activation_row = (activation_valid)? ub_data_o : 0;
    assign r_data = (r_from_ub)? ub_r_data : op_r_data;

    always_ff @(posedge clk or negedge rst_n) begin : ACTIVATION_MASK
        if(!rst_n) begin
            active_amask <= '1;
        end
        else if(amask_init) begin
            active_amask <= (amask_default_valid)? '1 : '0;
        end
        else if(amask_wr_en) begin
            active_amask[`A_BW*amask_wr_beat +: `A_BW] <= a_data;
        end
    end

    always_comb begin : ACTIVATION_MASK_ROW
        activation_mask_row = 0;
        if(activation_valid) begin
            for(int k = 0; k < `ARRAY_S; k++) begin
                activation_mask_row[k] =
                    active_amask[`ARRAY_S*activation_row_idx+k] &&
                    (k < k_valid_q);
            end
        end
    end

    generate
        for(genvar i = 0; i < `ARRAY_S; i++) begin : OUTPUT_FORMAT
            assign op_r_data[`R_W*i +: `R_W] = {op_data[`A_W*i +: `A_W], {(`R_W-`A_W){1'b0}}};
            assign ub_r_data[`R_W*i +: `R_W] = {ub_data_o[`A_W*i +: `A_W], {(`R_W-`A_W){1'b0}}};
        end
    endgenerate

    TSC u_TSC (
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
        .activation_row_idx(activation_row_idx),
        .amask_init(amask_init),
        .amask_default_valid(amask_default_valid),
        .amask_wr_en(amask_wr_en),
        .amask_wr_beat(amask_wr_beat),
        .bias_wr_en(bias_wr_en),
        .bias_wr_beat(bias_wr_beat),
        .acc_clr(acc_clr),
        .acc_bias_en(acc_bias_en),
        .acc_wr_en(acc_wr_en),
        .acc_wr_row(acc_wr_row),
        .acc_rd_en(acc_rd_en),
        .acc_rd_row(acc_rd_row),
        .k_valid_q(k_valid_q),
        .quant_mult_q(qmult_q),
        .quant_shift_q(qshift_q),
        .act_mode_q(act_mode_q)
    );

    WPU u_WPU (
        .clk(clk),
        .rst_n(rst_n),
        .weight(w_data),
        .wmem_addr_i(wmem_addr),
        .wmem_w(wpu_wmem_w),
        .rweight(wpu_rweight),
        .wmem_addr_o(wpu_wmem_addr)
    );

    WMEM_Wrapper u_WMEM_Wrapper (
        .clk(clk),
        .addr(wmem_mem_addr),
        .data_i(wpu_rweight),
        .en(wmem_en),
        .we(wmem_we),
        .data_o(wmem_data_o)
    );

    UB_Wrapper u_UB_Wrapper (
        .clk(clk),
        .addr(ub_addr),
        .data_i(ub_data_i),
        .en(ub_en),
        .we(ub_we),
        .data_o(ub_data_o)
    );

    Data_Setup u_Data_Setup (
        .clk(clk),
        .rst_n(rst_n),
        .activation_i(activation_row),
        .valid_i(activation_mask_row),
        .skew_en(skew_en),
        .activation_o(ds_activation_o),
        .valid_o(ds_activation_mask)
    );

    RSA u_RSA (
        .clk(clk),
        .rst_n(rst_n),
        .weight_valid(weight_valid),
        .weight(wmem_data_o),
        .activation(ds_activation_o),
        .activation_mask(ds_activation_mask),
        .psum(rsa_psum)
    );

    ACC u_ACC (
        .clk(clk),
        .rst_n(rst_n),
        .acc_clr(acc_clr),
        .acc_bias_en(acc_bias_en),
        .bias_wr_en(bias_wr_en),
        .bias_wr_beat(bias_wr_beat),
        .bias_data_i(w_data),
        .psum_i(rsa_psum),
        .acc_wr_en(acc_wr_en),
        .acc_wr_row(acc_wr_row),
        .acc_rd_en(acc_rd_en),
        .acc_rd_row(acc_rd_row),
        .acc_o(acc_o)
    );

    OP u_OP (
        .quant_mult(qmult_q),
        .quant_shift(qshift_q),
        .act_mode(act_mode_q),
        .acc_i(acc_o),
        .data_o(op_data)
    );

endmodule
