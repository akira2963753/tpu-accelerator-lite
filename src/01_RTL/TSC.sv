/******************************************************************************
* Copyright (C) 2026 Marco
*
* File Name:    TSC.sv
* Project:      tpu-accelerator-lite
* Module:       TSC (Tensor Processing Unit System Controller)
* Author:       Marco <harry2963753@gmail.com>
*
******************************************************************************/

module TSC (
    input clk,
    input rst_n,

    input cmd_valid,
    output logic cmd_ready,
    input [`CMD_DESC_W-1:0] cmd_desc,
    output logic busy,
    output logic done,

    input w_valid,
    output logic w_ready,
    input a_valid,
    output logic a_ready,
    output logic r_valid,
    output logic r_from_ub,
    input r_ready,

    output logic wpu_wmem_w,
    output logic [`WMEM_ADDR_W-1:0] wmem_addr,
    output logic wmem_en,
    output logic wmem_we,

    output logic [`UB_ADDR_W-1:0] ub_addr,
    output logic ub_en,
    output logic ub_we,

    output logic weight_valid,
    output logic skew_en,
    output logic activation_valid,
    output logic [`ROW_IDX_W-1:0] activation_row_idx,

    output logic amask_init,
    output logic amask_default_valid,
    output logic amask_wr_en,
    output logic [`AMASK_BEAT_W-1:0] amask_wr_beat,

    output logic bias_wr_en,
    output logic [`BIAS_BEAT_W-1:0] bias_wr_beat,

    output logic acc_clr,
    output logic acc_bias_en,
    output logic [`ARRAY_S-1:0] acc_wr_en,
    output logic [`ARRAY_S*`ROW_IDX_W-1:0] acc_wr_row,
    output logic acc_rd_en,
    output logic [`ROW_IDX_W-1:0] acc_rd_row,

    output logic [`K_VALID_W-1:0] k_valid_q,
    output logic [`QMULT_W-1:0] quant_mult_q,
    output logic [`QSHIFT_W-1:0] quant_shift_q,
    output logic [`ACT_MODE_W-1:0] act_mode_q
);

    localparam ACT_N = `ARRAY_S;
    localparam RESULT_LAT = `ARRAY_S + 1;
    localparam CAL_DRAIN = 2*`ARRAY_S + 4;
    localparam CAL_TOTAL = ACT_N + CAL_DRAIN;
    localparam TILE_CNT_W = $clog2(`ARRAY_S + 1);
    localparam GEMM_CNT_W = $clog2(CAL_TOTAL);

    typedef enum logic [3:0] {
        IDLE,
        LOAD_W,
        PRELOAD_W,
        LOAD_A,
        LOAD_BIAS,
        GEMM,
        STORE_C,
        READ_UB,
        DONE
    } state_t;

    state_t state, nx_state;

    wire [`CMD_OP_W-1:0] cmd_op;
    wire cmd_acc_init;
    wire cmd_bias_en;
    wire cmd_a_mask_en;
    wire [`K_VALID_W-1:0] cmd_k_valid;
    wire [`QMULT_W-1:0] cmd_quant_mult;
    wire [`QSHIFT_W-1:0] cmd_quant_shift;
    wire [`ACT_MODE_W-1:0] cmd_act_mode;
    wire [`WMEM_SLOT_W-1:0] cmd_w_slot;
    wire [`UB_SLOT_W-1:0] cmd_src_slot;
    wire [`UB_SLOT_W-1:0] cmd_dst_slot;
    wire cmd_accept;

    assign cmd_op = cmd_desc[`CMD_OP_LSB +: `CMD_OP_W];
    assign cmd_acc_init = cmd_desc[`CMD_ACC_INIT_BIT];
    assign cmd_bias_en = cmd_desc[`CMD_BIAS_EN_BIT];
    assign cmd_a_mask_en = cmd_desc[`CMD_A_MASK_EN_BIT];
    assign cmd_k_valid = cmd_desc[`CMD_K_VALID_LSB +: `K_VALID_W];
    assign cmd_quant_mult = cmd_desc[`CMD_QMULT_LSB +: `QMULT_W];
    assign cmd_quant_shift = cmd_desc[`CMD_QSHIFT_LSB +: `QSHIFT_W];
    assign cmd_act_mode = cmd_desc[`CMD_ACT_MODE_LSB +: `ACT_MODE_W];
    assign cmd_w_slot = cmd_desc[`CMD_W_SLOT_LSB +: `WMEM_SLOT_W];
    assign cmd_src_slot = cmd_desc[`CMD_SRC_SLOT_LSB +: `UB_SLOT_W];
    assign cmd_dst_slot = cmd_desc[`CMD_DST_SLOT_LSB +: `UB_SLOT_W];
    assign cmd_accept = (state == IDLE) && cmd_valid;

    logic acc_init_q;
    logic bias_en_q;
    logic a_mask_en_q;
    logic [`WMEM_SLOT_W-1:0] w_slot_q;
    logic [`UB_SLOT_W-1:0] src_slot_q;
    logic [`UB_SLOT_W-1:0] dst_slot_q;

    logic [TILE_CNT_W-1:0] a_cnt;
    logic [TILE_CNT_W-1:0] w_cnt;
    logic [`BIAS_BEAT_W:0] bias_cnt;
    logic [TILE_CNT_W-1:0] pw;
    logic [GEMM_CNT_W-1:0] cc;
    logic [TILE_CNT_W-1:0] ov;
    logic [TILE_CNT_W-1:0] ru_cnt;
    logic ru_valid_q;

    wire a_beat;
    wire w_beat;
    wire bias_beat;
    wire ru_issue;
    wire [TILE_CNT_W-1:0] a_limit;

    assign a_limit = (a_mask_en_q)? (`ARRAY_S + `AMASK_BEATS) : `ARRAY_S;
    assign a_beat = (state == LOAD_A) && a_valid && (a_cnt != a_limit);
    assign w_beat = (state == LOAD_W) && w_valid && (w_cnt != `ARRAY_S);
    assign bias_beat = (state == LOAD_BIAS) && w_valid &&
                       (bias_cnt != `BIAS_BEATS);
    assign ru_issue = (state == READ_UB) &&
                      (ru_cnt < `ARRAY_S) &&
                      (!ru_valid_q || r_ready);

    logic en_chain [0:`ARRAY_S-1];
    logic [`ROW_IDX_W-1:0] row_chain [0:`ARRAY_S-1];
    wire res_valid;
    wire [`ROW_IDX_W-1:0] res_row;

    assign res_valid = (state == GEMM) &&
                       (cc >= RESULT_LAT) &&
                       (cc < RESULT_LAT + ACT_N);
    assign res_row = cc - RESULT_LAT;

    logic wmem_wr_q;

    always_comb begin : NEXT_STATE
        case(state)
            IDLE: begin
                if(cmd_valid) begin
                    case(cmd_op)
                        `CMD_OP_LOAD_W: nx_state = LOAD_W;
                        `CMD_OP_PRELOAD_W: nx_state = PRELOAD_W;
                        `CMD_OP_LOAD_A: nx_state = LOAD_A;
                        `CMD_OP_LOAD_BIAS: nx_state = LOAD_BIAS;
                        `CMD_OP_GEMM: nx_state = GEMM;
                        `CMD_OP_STORE_C: nx_state = STORE_C;
                        `CMD_OP_READ_UB: nx_state = READ_UB;
                        default: nx_state = DONE;
                    endcase
                end
                else nx_state = IDLE;
            end
            LOAD_W: begin
                if(w_cnt == `ARRAY_S) nx_state = DONE;
                else nx_state = LOAD_W;
            end
            PRELOAD_W: begin
                if(pw == `ARRAY_S) nx_state = DONE;
                else nx_state = PRELOAD_W;
            end
            LOAD_A: begin
                if(a_cnt == a_limit) nx_state = DONE;
                else nx_state = LOAD_A;
            end
            LOAD_BIAS: begin
                if(bias_cnt == `BIAS_BEATS) nx_state = DONE;
                else nx_state = LOAD_BIAS;
            end
            GEMM: begin
                if(cc == CAL_TOTAL-1) nx_state = DONE;
                else nx_state = GEMM;
            end
            STORE_C: begin
                if(r_ready && ov == `ARRAY_S-1) nx_state = DONE;
                else nx_state = STORE_C;
            end
            READ_UB: begin
                if(ru_valid_q && r_ready && ru_cnt == `ARRAY_S) nx_state = DONE;
                else nx_state = READ_UB;
            end
            DONE: nx_state = IDLE;
            default: nx_state = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin : STATE_REG
        if(!rst_n) state <= IDLE;
        else state <= nx_state;
    end

    always_ff @(posedge clk or negedge rst_n) begin : CONTROL_REG
        if(!rst_n) begin
            acc_init_q <= 0;
            bias_en_q <= 0;
            a_mask_en_q <= 0;
            w_slot_q <= 0;
            src_slot_q <= 0;
            dst_slot_q <= 0;
            quant_mult_q <= 0;
            quant_shift_q <= 0;
            k_valid_q <= 0;
            act_mode_q <= `ACT_NONE;
            a_cnt <= 0;
            w_cnt <= 0;
            bias_cnt <= 0;
            pw <= 0;
            cc <= 0;
            ov <= 0;
            ru_cnt <= 0;
            ru_valid_q <= 0;
            wmem_wr_q <= 0;
        end
        else begin
            if(cmd_accept) begin
                acc_init_q <= cmd_acc_init;
                bias_en_q <= cmd_bias_en;
                a_mask_en_q <= cmd_a_mask_en;
                w_slot_q <= cmd_w_slot;
                src_slot_q <= cmd_src_slot;
                dst_slot_q <= cmd_dst_slot;
                quant_mult_q <= cmd_quant_mult;
                quant_shift_q <= cmd_quant_shift;
                k_valid_q <= cmd_k_valid;
                act_mode_q <= cmd_act_mode;
            end

            if(state == LOAD_W) begin
                if(w_beat) w_cnt <= w_cnt + 1'b1;
            end
            else begin
                w_cnt <= 0;
            end

            if(state == LOAD_A) begin
                if(a_beat) a_cnt <= a_cnt + 1'b1;
            end
            else begin
                a_cnt <= 0;
            end

            if(state == LOAD_BIAS) begin
                if(bias_beat) bias_cnt <= bias_cnt + 1'b1;
            end
            else begin
                bias_cnt <= 0;
            end

            if(state == PRELOAD_W) begin
                if(pw == `ARRAY_S) pw <= 0;
                else pw <= pw + 1'b1;
            end
            else begin
                pw <= 0;
            end

            if(state == GEMM) begin
                if(cc == CAL_TOTAL-1) cc <= 0;
                else cc <= cc + 1'b1;
            end
            else begin
                cc <= 0;
            end

            if(state == STORE_C) begin
                if(r_ready) begin
                    if(ov == `ARRAY_S-1) ov <= 0;
                    else ov <= ov + 1'b1;
                end
            end
            else begin
                ov <= 0;
            end

            if(state == READ_UB) begin
                if(ru_issue) ru_cnt <= ru_cnt + 1'b1;

                if(ru_issue) ru_valid_q <= 1'b1;
                else if(ru_valid_q && r_ready) ru_valid_q <= 1'b0;
            end
            else begin
                ru_cnt <= 0;
                ru_valid_q <= 0;
            end

            wmem_wr_q <= w_beat;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin : RESULT_PIPE
        if(!rst_n) begin
            for(int j = 0; j < `ARRAY_S; j++) begin
                en_chain[j] <= 0;
                row_chain[j] <= 0;
            end
        end
        else begin
            en_chain[0] <= res_valid;
            row_chain[0] <= res_row;
            for(int j = 1; j < `ARRAY_S; j++) begin
                en_chain[j] <= en_chain[j-1];
                row_chain[j] <= row_chain[j-1];
            end
        end
    end

    always_comb begin : CONTROL_OUTPUT
        cmd_ready = (state == IDLE);
        busy = (state != IDLE);
        done = (state == DONE);

        w_ready = ((state == LOAD_W) && (w_cnt != `ARRAY_S)) ||
                  ((state == LOAD_BIAS) && (bias_cnt != `BIAS_BEATS));
        a_ready = (state == LOAD_A) && (a_cnt != a_limit);
        r_valid = (state == STORE_C) || ((state == READ_UB) && ru_valid_q);
        r_from_ub = (state == READ_UB);

        amask_init = cmd_accept && (cmd_op == `CMD_OP_LOAD_A);
        amask_default_valid = !cmd_a_mask_en;
        amask_wr_en = (state == LOAD_A) && a_beat &&
                      (a_cnt >= `ARRAY_S);
        amask_wr_beat = 0;
        if(a_cnt >= `ARRAY_S) amask_wr_beat = a_cnt - `ARRAY_S;

        bias_wr_en = bias_beat;
        bias_wr_beat = bias_cnt[`BIAS_BEAT_W-1:0];

        wpu_wmem_w = w_beat;
        wmem_addr = 0;
        wmem_en = wmem_wr_q;
        wmem_we = wmem_wr_q;

        if(state == LOAD_W) begin
            wmem_addr = {w_slot_q, w_cnt[`ROW_IDX_W-1:0]};
        end
        else if(state == PRELOAD_W && pw < `ARRAY_S) begin
            wmem_addr = {w_slot_q, `ROW_IDX_W'((`ARRAY_S-1)-pw)};
            wmem_en = 1'b1;
            wmem_we = 1'b0;
        end

        ub_addr = 0;
        ub_en = 0;
        ub_we = 0;

        if(state == LOAD_A && a_beat && a_cnt < `ARRAY_S) begin
            ub_addr = {src_slot_q, a_cnt[`ROW_IDX_W-1:0]};
            ub_en = 1'b1;
            ub_we = 1'b1;
        end
        else if(state == GEMM && cc < ACT_N) begin
            ub_addr = {src_slot_q, cc[`ROW_IDX_W-1:0]};
            ub_en = 1'b1;
        end
        else if(state == STORE_C && r_ready) begin
            ub_addr = {dst_slot_q, ov[`ROW_IDX_W-1:0]};
            ub_en = 1'b1;
            ub_we = 1'b1;
        end
        else if(state == READ_UB && ru_issue) begin
            ub_addr = {src_slot_q, ru_cnt[`ROW_IDX_W-1:0]};
            ub_en = 1'b1;
        end

        weight_valid = (state == PRELOAD_W) && (pw > 0) && (pw <= `ARRAY_S);
        skew_en = (state == GEMM) && (cc >= 1);
        activation_valid = (state == GEMM) && (cc >= 1) && (cc <= ACT_N);
        activation_row_idx = (activation_valid)? cc - 1'b1 : 0;

        acc_clr = (state == GEMM) && acc_init_q && (cc == 0);
        acc_bias_en = acc_init_q && bias_en_q;

        for(int j = 0; j < `ARRAY_S; j++) begin
            acc_wr_en[j] = en_chain[j];
            acc_wr_row[`ROW_IDX_W*j +: `ROW_IDX_W] = row_chain[j];
        end

        acc_rd_en = (state == STORE_C);
        acc_rd_row = ov[`ROW_IDX_W-1:0];
    end

endmodule
